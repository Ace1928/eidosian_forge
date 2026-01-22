import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MegaConfig
    base_model_prefix = 'mega'
    supports_gradient_checkpointing = False
    _no_split_modules = ['MegaMovingAverageGatedAttention']

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, MegaMultiDimensionDampedEma):
            with torch.no_grad():
                nn.init.normal_(module.damping_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                nn.init.normal_(module.decay_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                val = torch.ones(self.config.ema_projection_size, 1)
                if self.config.ema_projection_size > 1:
                    idx = torch.tensor(list(range(1, self.config.ema_projection_size, 2)))
                    val.index_fill_(0, idx, -1.0)
                module.ema_expansion_matrix.normal_(mean=0.0, std=self.config.ema_beta_range).add_(val)
                nn.init.normal_(module.kernel_projection_matrix, mean=0.0, std=self.config.ema_gamma_omega_range)
                nn.init.normal_(module.residual_weight, mean=0.0, std=self.config.ema_gamma_omega_range)
        elif isinstance(module, MegaSimpleRelativePositionalBias):
            nn.init.normal_(module.rel_pos_bias, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MegaRotaryRelativePositionalBias):
            nn.init.normal_(module.alpha, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.b_param, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MegaScaleNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.scalar, 1.0)
        elif isinstance(module, MegaRMSNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, MegaMovingAverageGatedAttention):
            nn.init.normal_(module.qk_weight, mean=0.0, std=self.config.initializer_range)
            nn.init.constant_(module.qk_bias, 0.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)