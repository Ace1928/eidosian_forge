import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer
from typing import Optional, List, Callable, Union, Any, cast
def _enable_tensor_cores():
    global _warned_tensor_cores
    if torch.cuda.is_available():
        if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
            torch.set_float32_matmul_precision('high')
            if not _warned_tensor_cores:
                print('Your GPU supports tensor cores')
                print("we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
                _warned_tensor_cores = True