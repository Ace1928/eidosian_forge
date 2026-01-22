import warnings
from typing import Callable, Dict, List, Optional, Union
from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from .. import mobilenet, resnet
from .._api import _get_enum_from_fn, WeightsEnum
from .._utils import handle_legacy_interface, IntermediateLayerGetter
def _mobilenet_extractor(backbone: Union[mobilenet.MobileNetV2, mobilenet.MobileNetV3], fpn: bool, trainable_layers: int, returned_layers: Optional[List[int]]=None, extra_blocks: Optional[ExtraFPNBlock]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> nn.Module:
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, '_is_cn', False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f'Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ')
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]
    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)
    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
            raise ValueError(f'Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ')
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)
    else:
        m = nn.Sequential(backbone, nn.Conv2d(backbone[-1].out_channels, out_channels, 1))
        m.out_channels = out_channels
        return m