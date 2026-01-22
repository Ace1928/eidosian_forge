from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
    """

    def __init__(self, block_size: int, gpu_memory_utilization: float, swap_space: int, cache_dtype: str, sliding_window: Optional[int]=None) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.cache_dtype = cache_dtype
        self.sliding_window = sliding_window
        self._verify_args()
        self._verify_cache_dtype()
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def metrics_info(self):
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(f'GPU memory utilization must be less than 1.0. Got {self.gpu_memory_utilization}.')

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == 'auto':
            pass
        elif self.cache_dtype == 'fp8_e5m2':
            nvcc_cuda_version = get_nvcc_cuda_version()
            if nvcc_cuda_version and nvcc_cuda_version < Version('11.8'):
                raise ValueError('FP8 is not supported when cuda version is lower than 11.8.')
            device_name = torch.cuda.get_device_name()
            if 'AMD' in device_name:
                raise NotImplementedError('FP8_E5M2 KV Cache on AMD GPU has not been supported yet.')
            logger.info('Using fp8_e5m2 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. But it may cause slight accuracy drop. Currently we only support fp8 without scaling factors and make e5m2 as a default format.')
        else:
            raise ValueError(f'Unknown kv cache dtype: {self.cache_dtype}')

    def verify_with_parallel_config(self, parallel_config: 'ParallelConfig') -> None:
        total_cpu_memory = get_cpu_memory()
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node
        msg = f'{cpu_memory_usage / _GB:.2f} GiB out of the {total_cpu_memory / _GB:.2f} GiB total CPU memory is allocated for the swap space.'
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError('Too large swap space. ' + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning('Possibly too large swap space. ' + msg)