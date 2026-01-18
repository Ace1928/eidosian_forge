import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
check if CuPy nccl is enabled for all reduce