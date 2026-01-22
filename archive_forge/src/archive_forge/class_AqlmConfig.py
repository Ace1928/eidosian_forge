import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
@dataclass
class AqlmConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `aqlm` parameters.

    Args:
        in_group_size (`int`, *optional*, defaults to 8):
            The group size along the input dimension.
        out_group_size (`int`, *optional*, defaults to 1):
            The group size along the output dimension. It's recommended to always use 1.
        num_codebooks (`int`, *optional*, defaults to 1):
            Number of codebooks for the Additive Quantization procedure.
        nbits_per_codebook (`int`, *optional*, defaults to 16):
            Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
        linear_weights_not_to_quantize (`Optional[List[str]]`, *optional*):
            List of full paths of `nn.Linear` weight parameters that shall not be quantized.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(self, in_group_size: int=8, out_group_size: int=1, num_codebooks: int=1, nbits_per_codebook: int=16, linear_weights_not_to_quantize: Optional[List[str]]=None, **kwargs):
        self.quant_method = QuantizationMethod.AQLM
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.linear_weights_not_to_quantize = linear_weights_not_to_quantize
        self.post_init()

    def post_init(self):
        """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.in_group_size, int):
            raise ValueError('in_group_size must be a float')
        if not isinstance(self.out_group_size, int):
            raise ValueError('out_group_size must be a float')
        if not isinstance(self.num_codebooks, int):
            raise ValueError('num_codebooks must be a float')
        if not isinstance(self.nbits_per_codebook, int):
            raise ValueError('nbits_per_codebook must be a float')
        if self.linear_weights_not_to_quantize is not None and (not isinstance(self.linear_weights_not_to_quantize, list)):
            raise ValueError('linear_weights_not_to_quantize must be a list of strings')
        if self.linear_weights_not_to_quantize is None:
            self.linear_weights_not_to_quantize = []