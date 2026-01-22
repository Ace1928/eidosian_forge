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
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
@add_start_docstrings('\n    The bare Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) outputting raw\n    hidden-states without any specific head on top.\n    ', GROUNDING_DINO_START_DOCSTRING)
class GroundingDinoModel(GroundingDinoPreTrainedModel):

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)
        backbone = GroundingDinoConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = GroundingDinoConvModel(backbone, position_embeddings)
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[i]
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, config.d_model, kernel_size=1), nn.GroupNorm(32, config.d_model)))
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, config.d_model)))
                in_channels = config.d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList([nn.Sequential(nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1), nn.GroupNorm(32, config.d_model))])
        self.text_backbone = AutoModel.from_config(config.text_config, add_pooling_layer=False)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)
        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.encoder = GroundingDinoEncoder(config)
        self.decoder = GroundingDinoDecoder(config)
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
            if config.two_stage_bbox_embed_share and config.decoder_bbox_embed_share and (self.decoder.bbox_embed is not None):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3)
            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
        else:
            self.reference_points = nn.Embedding(config.num_queries, 4)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def generate_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (`torch.Tensor[batch_size, sequence_length, hidden_size]`): Output of the encoder.
            padding_mask (`torch.Tensor[batch_size, sequence_length]`): Padding mask for `enc_output`.
            spatial_shapes (`torch.Tensor[num_feature_levels, 2]`): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, current_position:current_position + height * width]
            mask_flatten_ = mask_flatten_.view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = meshgrid(torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device), torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device), indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * 2.0 ** level
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current_position += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return (object_query, output_proposals)

    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, input_ids: Tensor, token_type_ids: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, pixel_mask: Optional[Tensor]=None, encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        text_token_mask = attention_mask.bool()
        max_text_len = self.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]
        text_outputs = self.text_backbone(input_ids, text_self_attention_masks, token_type_ids, position_ids, return_dict=return_dict)
        text_features = text_outputs.last_hidden_state if return_dict else text_outputs[0]
        text_features = self.text_projection(text_features)
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)
        vision_features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(vision_features):
            feature_maps.append(self.input_proj_vision[level](source))
            masks.append(mask)
        if self.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj_vision[level](vision_features[-1][0])
                else:
                    source = self.input_proj_vision[level](feature_maps[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()
        if encoder_outputs is None:
            encoder_outputs = self.encoder(vision_features=source_flatten, vision_attention_mask=~mask_flatten, vision_position_embedding=lvl_pos_embed_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, text_features=text_features, text_attention_mask=~text_token_mask, text_position_embedding=None, text_self_attention_masks=~text_self_attention_masks, text_position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, GroundingDinoEncoderOutput)):
            encoder_outputs = GroundingDinoEncoderOutput(last_hidden_state_vision=encoder_outputs[0], last_hidden_state_text=encoder_outputs[1], vision_hidden_states=encoder_outputs[2] if output_hidden_states else None, text_hidden_states=encoder_outputs[3] if output_hidden_states else None, attentions=encoder_outputs[-1] if output_attentions else None)
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.generate_encoder_output_proposals(encoder_outputs[0], ~mask_flatten, spatial_shapes)
            enc_outputs_class = self.encoder_output_class_embed(object_query_embedding, encoder_outputs[1], text_token_mask)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)).detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_reference_points = reference_points
        decoder_outputs = self.decoder(inputs_embeds=target, vision_encoder_hidden_states=encoder_outputs[0], vision_encoder_attention_mask=mask_flatten, text_encoder_hidden_states=encoder_outputs[1], text_encoder_attention_mask=~text_token_mask, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, self_attn_mask=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            enc_outputs = tuple((value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None))
            tuple_outputs = (decoder_outputs[0], init_reference_points) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            return tuple_outputs
        return GroundingDinoModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, init_reference_points=init_reference_points, intermediate_hidden_states=decoder_outputs.intermediate_hidden_states, intermediate_reference_points=decoder_outputs.intermediate_reference_points, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, encoder_last_hidden_state_vision=encoder_outputs.last_hidden_state_vision, encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text, encoder_vision_hidden_states=encoder_outputs.vision_hidden_states, encoder_text_hidden_states=encoder_outputs.text_hidden_states, encoder_attentions=encoder_outputs.attentions, enc_outputs_class=enc_outputs_class, enc_outputs_coord_logits=enc_outputs_coord_logits)