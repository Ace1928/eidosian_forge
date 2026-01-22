from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils import is_hip
class SqueezeLLMConfig(QuantizationConfig):
    """Config class for SqueezeLLM.

    Reference: https://arxiv.org/pdf/2306.07629
    """

    def __init__(self, weight_bits: int) -> None:
        self.weight_bits = weight_bits
        if self.weight_bits != 4:
            raise ValueError(f'Currently, only 4-bit weight quantization is supported for SqueezeLLM, but got {self.weight_bits} bits.')
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return f'SqueezeLLMConfig(weight_bits={self.weight_bits})'

    def get_name(self) -> str:
        return 'squeezellm'

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ['quant_config.json']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SqueezeLLMConfig':
        weight_bits = cls.get_from_keys(config, ['wbits'])
        return cls(weight_bits)

    def get_linear_method(self) -> 'SqueezeLLMLinearMethod':
        return SqueezeLLMLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []