import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
@add_start_docstrings('\n    The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without\n    any specific head on top.\n    ', DETA_START_DOCSTRING)
class DetaModel(DetaPreTrainedModel):

    def __init__(self, config: DetaConfig):
        super().__init__(config)
        if config.two_stage:
            requires_backends(self, ['torchvision'])
        self.backbone = DetaBackboneWithPositionalEncodings(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes
        if config.num_feature_levels > 1:
            num_backbone_outs = len(intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = intermediate_channel_sizes[_]
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, config.d_model, kernel_size=1), nn.GroupNorm(32, config.d_model)))
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, config.d_model)))
                in_channels = config.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(intermediate_channel_sizes[-1], config.d_model, kernel_size=1), nn.GroupNorm(32, config.d_model))])
        if not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)
        self.encoder = DetaEncoder(config)
        self.decoder = DetaDecoder(config)
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model)
            self.pos_trans = nn.Linear(config.d_model * 2, config.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(config.d_model * 2)
            self.pix_trans = nn.Linear(config.d_model, config.d_model)
            self.pix_trans_norm = nn.LayerNorm(config.d_model)
        else:
            self.reference_points = nn.Linear(config.d_model, 2)
        self.assign_first_stage = config.assign_first_stage
        self.two_stage_num_proposals = config.two_stage_num_proposals
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals):
        """Get the position embedding of the proposals."""
        num_pos_feats = self.config.d_model // 2
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.int64, device=proposals.device).float()
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        level_ids = []
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur:_cur + height * width].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = meshgrid(torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device), torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device), indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * 2.0 ** level
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
            level_ids.append(grid.new_ones(height * width, dtype=torch.long) * level)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        level_ids = torch.cat(level_ids)
        return (object_query, output_proposals, level_ids)

    @add_start_docstrings_to_model_forward(DETA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetaModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], DetaModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetaModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large-o365")
        >>> model = DetaModel.from_pretrained("jozhang97/deta-swin-large-o365", two_stage=False)

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError('No attention mask was provided')
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
        query_embeds = None
        if not self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight
        spatial_shapes = [source.shape[2:] for source in sources]
        source_flatten = [source.flatten(2).transpose(1, 2) for source in sources]
        mask_flatten = [mask.flatten(1) for mask in masks]
        lvl_pos_embed_flatten = []
        for level, pos_embed in enumerate(position_embeddings_list):
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=source_flatten, attention_mask=mask_flatten, position_embeddings=lvl_pos_embed_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        output_proposals = None
        if self.config.two_stage:
            object_query_embedding, output_proposals, level_ids = self.gen_encoder_output_proposals(encoder_outputs[0], ~mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[-1](object_query_embedding)
            delta_bbox = self.decoder.bbox_embed[-1](object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals
            topk = self.two_stage_num_proposals
            proposal_logit = enc_outputs_class[..., 0]
            if self.assign_first_stage:
                proposal_boxes = center_to_corners_format(enc_outputs_coord_logits.sigmoid().float()).clamp(0, 1)
                topk_proposals = []
                for b in range(batch_size):
                    prop_boxes_b = proposal_boxes[b]
                    prop_logits_b = proposal_logit[b]
                    pre_nms_topk = 1000
                    pre_nms_inds = []
                    for lvl in range(len(spatial_shapes)):
                        lvl_mask = level_ids == lvl
                        pre_nms_inds.append(torch.topk(prop_logits_b.sigmoid() * lvl_mask, pre_nms_topk)[1])
                    pre_nms_inds = torch.cat(pre_nms_inds)
                    post_nms_inds = batched_nms(prop_boxes_b[pre_nms_inds], prop_logits_b[pre_nms_inds], level_ids[pre_nms_inds], 0.9)
                    keep_inds = pre_nms_inds[post_nms_inds]
                    if len(keep_inds) < self.two_stage_num_proposals:
                        print(f'[WARNING] nms proposals ({len(keep_inds)}) < {self.two_stage_num_proposals}, running naive topk')
                        keep_inds = torch.topk(proposal_logit[b], topk)[1]
                    q_per_l = topk // len(spatial_shapes)
                    is_level_ordered = level_ids[keep_inds][None] == torch.arange(len(spatial_shapes), device=level_ids.device)[:, None]
                    keep_inds_mask = is_level_ordered & (is_level_ordered.cumsum(1) <= q_per_l)
                    keep_inds_mask = keep_inds_mask.any(0)
                    if keep_inds_mask.sum() < topk:
                        num_to_add = topk - keep_inds_mask.sum()
                        pad_inds = (~keep_inds_mask).nonzero()[:num_to_add]
                        keep_inds_mask[pad_inds] = True
                    keep_inds_topk = keep_inds[keep_inds_mask]
                    topk_proposals.append(keep_inds_topk)
                topk_proposals = torch.stack(topk_proposals)
            else:
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_logits = torch.gather(enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_logits)))
            query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)
            topk_feats = torch.stack([object_query_embedding[b][topk_proposals[b]] for b in range(batch_size)]).detach()
            target = target + self.pix_trans_norm(self.pix_trans(topk_feats))
        else:
            query_embed, target = torch.split(query_embeds, num_channels, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_points = reference_points
        decoder_outputs = self.decoder(inputs_embeds=target, position_embeddings=query_embed, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            enc_outputs = tuple((value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None))
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + enc_outputs
            return tuple_outputs
        return DetaModelOutput(init_reference_points=init_reference_points, last_hidden_state=decoder_outputs.last_hidden_state, intermediate_hidden_states=decoder_outputs.intermediate_hidden_states, intermediate_reference_points=decoder_outputs.intermediate_reference_points, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, enc_outputs_class=enc_outputs_class, enc_outputs_coord_logits=enc_outputs_coord_logits, output_proposals=output_proposals)