import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_0
from ...utils import (
from .configuration_falcon import FalconConfig
class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FalconConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['FalconDecoderLayer']
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, FalconLinear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool=False) -> 'PretrainedConfig':
        if hard_check_only:
            if not is_torch_greater_or_equal_than_2_0:
                raise ImportError('PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.0.')
        if not is_torch_greater_or_equal_than_2_0:
            return config
        _is_bettertransformer = getattr(cls, 'use_bettertransformer', False)
        if _is_bettertransformer:
            return config
        if not hard_check_only:
            config._attn_implementation = 'sdpa'
        return config