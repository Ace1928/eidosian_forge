from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
class MarlinConfig(QuantizationConfig):
    """Config class for Marlin.

    Reference: https://github.com/IST-DASLab/marlin/tree/master
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        if self.group_size != 128 and self.group_size != -1:
            raise ValueError(f'Currently, only group size 128 and -1 (channelwise) is supported for Marlin, but got group_size of {self.group_size}')
        self.pack_factor = 32 // 4
        self.tile_size = 16
        self.min_n_threads = 64
        self.min_k_threads = 128
        self.max_parallel = 16
        self.perm_len = 1024

    def __repr__(self) -> str:
        return f'MarlinConfig(group_size={self.group_size}'

    @classmethod
    def get_name(cls) -> str:
        return 'marlin'

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ['quantize_config.json']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MarlinConfig':
        group_size = cls.get_from_keys(config, ['group_size'])
        return cls(group_size)

    def get_linear_method(self) -> 'MarlinLinearMethod':
        return MarlinLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []