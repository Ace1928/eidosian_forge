from typing import Dict, List, Tuple
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, is_neuron, STR_DTYPE_TO_TORCH_DTYPE
def allocate_gpu_cache(self) -> List[KVCache]:
    gpu_cache: List[KVCache] = []
    key_block_shape = self.get_key_block_shape()
    value_block_shape = self.get_value_block_shape()
    for _ in range(self.num_layers):
        key_blocks = torch.empty(size=(self.num_gpu_blocks, *key_block_shape), dtype=self.dtype, device='cuda')
        value_blocks = torch.empty(size=(self.num_gpu_blocks, *value_block_shape), dtype=self.dtype, device='cuda')
        gpu_cache.append((key_blocks, value_blocks))
    return gpu_cache