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