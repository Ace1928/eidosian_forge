from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class RAFT(nn.Module):

    def __init__(self, *, feature_encoder, context_encoder, corr_block, update_block, mask_predictor=None):
        """RAFT model from
        `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`_.

        args:
            feature_encoder (nn.Module): The feature encoder. It must downsample the input by 8.
                Its input is the concatenation of ``image1`` and ``image2``.
            context_encoder (nn.Module): The context encoder. It must downsample the input by 8.
                Its input is ``image1``. As in the original implementation, its output will be split into 2 parts:

                - one part will be used as the actual "context", passed to the recurrent unit of the ``update_block``
                - one part will be used to initialize the hidden state of the recurrent unit of
                  the ``update_block``

                These 2 parts are split according to the ``hidden_state_size`` of the ``update_block``, so the output
                of the ``context_encoder`` must be strictly greater than ``hidden_state_size``.

            corr_block (nn.Module): The correlation block, which creates a correlation pyramid from the output of the
                ``feature_encoder``, and then indexes from this pyramid to create correlation features. It must expose
                2 methods:

                - a ``build_pyramid`` method that takes ``feature_map_1`` and ``feature_map_2`` as input (these are the
                  output of the ``feature_encoder``).
                - a ``index_pyramid`` method that takes the coordinates of the centroid pixels as input, and returns
                  the correlation features. See paper section 3.2.

                It must expose an ``out_channels`` attribute.

            update_block (nn.Module): The update block, which contains the motion encoder, the recurrent unit, and the
                flow head. It takes as input the hidden state of its recurrent unit, the context, the correlation
                features, and the current predicted flow. It outputs an updated hidden state, and the ``delta_flow``
                prediction (see paper appendix A). It must expose a ``hidden_state_size`` attribute.
            mask_predictor (nn.Module, optional): Predicts the mask that will be used to upsample the predicted flow.
                The output channel must be 8 * 8 * 9 - see paper section 3.3, and Appendix B.
                If ``None`` (default), the flow is upsampled using interpolation.
        """
        super().__init__()
        _log_api_usage_once(self)
        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block
        self.mask_predictor = mask_predictor
        if not hasattr(self.update_block, 'hidden_state_size'):
            raise ValueError("The update_block parameter should expose a 'hidden_state_size' attribute.")

    def forward(self, image1, image2, num_flow_updates: int=12):
        batch_size, _, h, w = image1.shape
        if (h, w) != image2.shape[-2:]:
            raise ValueError(f'input images should have the same shape, instead got ({h}, {w}) != {image2.shape[-2:]}')
        if not h % 8 == 0 and w % 8 == 0:
            raise ValueError(f'input image H and W should be divisible by 8, instead got {h} (h) and {w} (w)')
        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        if fmap1.shape[-2:] != (h // 8, w // 8):
            raise ValueError('The feature encoder should downsample H and W by 8')
        self.corr_block.build_pyramid(fmap1, fmap2)
        context_out = self.context_encoder(image1)
        if context_out.shape[-2:] != (h // 8, w // 8):
            raise ValueError('The context encoder should downsample H and W by 8')
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size
        if out_channels_context <= 0:
            raise ValueError(f'The context encoder outputs {context_out.shape[1]} channels, but it should have at strictly more than hidden_state={hidden_state_size} channels')
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F.relu(context)
        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)
            coords1 = coords1 + delta_flow
            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            upsampled_flow = upsample_flow(flow=coords1 - coords0, up_mask=up_mask)
            flow_predictions.append(upsampled_flow)
        return flow_predictions