from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_opt import OPTConfig
def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
    """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
    val = None
    if fn_arg_name in kwargs:
        logging.warning("Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38. Please set it in the config instead")
        val = kwargs.pop(fn_arg_name)
    else:
        val = getattr(config, config_arg_name)
    return val