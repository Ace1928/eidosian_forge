import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@classmethod
def _check_and_enable_flash_attn_2(cls, config, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None, hard_check_only: bool=False):
    """
        `_check_and_enable_flash_attn_2` originally don't expand flash attention enabling to the model
        sub-configurations. We override the original method to make sure that Bark sub-models are using Flash Attention
        if necessary.

        If you don't know about Flash Attention, check out the official repository of flash attention:
        https://github.com/Dao-AILab/flash-attention

        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this
        specific section of the documentation to learn more about it:
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models

        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in
        half precision and not ran on CPU.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flash_attention_2" so that the model
        can initialize the correct attention module
        """
    config = super()._check_and_enable_flash_attn_2(config, torch_dtype, device_map, hard_check_only=hard_check_only)
    config.semantic_config._attn_implementation = config._attn_implementation
    config.coarse_acoustics_config._attn_implementation = config._attn_implementation
    config.fine_acoustics_config._attn_implementation = config._attn_implementation
    return config