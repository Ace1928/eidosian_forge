import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerPreTrainedModel(PreTrainedModel):
    config_class = MaskFormerConfig
    base_model_prefix = 'model'
    main_input_name = 'pixel_values'

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(module, MaskFormerTransformerModule):
            if module.input_projection is not None:
                nn.init.xavier_uniform_(module.input_projection.weight, gain=xavier_std)
                nn.init.constant_(module.input_projection.bias, 0)
        elif isinstance(module, MaskFormerFPNModel):
            nn.init.xavier_uniform_(module.stem.get_submodule('0').weight, gain=xavier_std)
        elif isinstance(module, MaskFormerFPNLayer):
            nn.init.xavier_uniform_(module.proj[0].weight, gain=xavier_std)
        elif isinstance(module, MaskFormerFPNConvLayer):
            nn.init.xavier_uniform_(module.get_submodule('0').weight, gain=xavier_std)
        elif isinstance(module, MaskformerMLPPredictionHead):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=xavier_std)
                    nn.init.constant_(submodule.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()