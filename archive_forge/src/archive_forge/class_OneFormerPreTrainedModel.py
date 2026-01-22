import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerPreTrainedModel(PreTrainedModel):
    config_class = OneFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(module, OneFormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)
        elif isinstance(module, OneFormerTransformerDecoder):
            nn.init.xavier_uniform_(module.query_input_projection.weight, gain=xavier_std)
            nn.init.constant_(module.query_input_projection.bias, 0)
            module.query_input_projection._is_hf_initialized = True
        elif isinstance(module, OneFormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(module.n_heads, 1, 1, 2).repeat(1, module.n_levels, module.n_points, 1)
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, OneFormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif isinstance(module, OneFormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)
        elif isinstance(module, OneFormerTransformerDecoderSelfAttentionLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderCrossAttentionLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderFFNLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerTransformerDecoderQueryTransformer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, OneFormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()
        elif isinstance(module, OneFormerTextContextDecoder):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.trunc_normal_(submodule.weight, std=0.02)
                    if isinstance(submodule, nn.Linear) and submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.constant_(submodule.bias, 0)
                    nn.init.constant_(submodule.weight, 1.0)
        elif isinstance(module, OneFormerTextTransformer):
            proj_std = module.width ** (-0.5) * (2 * module.num_layers) ** (-0.5)
            attn_std = module.width ** (-0.5)
            fc_std = (2 * module.width) ** (-0.5)
            for layer in module.layers:
                nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
                nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
                nn.init.normal_(layer.mlp.fc1.weight, std=fc_std)
                nn.init.normal_(layer.mlp.fc2.weight, std=proj_std)
        elif isinstance(module, OneFormerTextEncoder):
            nn.init.normal_(module.token_embedding.weight, std=0.02)
            nn.init.normal_(module.positional_embedding, std=0.01)
        if hasattr(module, 'reference_points'):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        elif isinstance(module, OneFormerTaskModel):
            for submodule in module.modules():
                if isinstance(module, OneFormerMLPPredictionHead):
                    for submodule in module.modules():
                        if isinstance(submodule, nn.Linear):
                            nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                            nn.init.constant_(submodule.bias, 0)
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=std)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()