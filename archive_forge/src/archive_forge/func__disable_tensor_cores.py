import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer
from typing import Optional, List, Callable, Union, Any, cast
def _disable_tensor_cores():
    torch.set_float32_matmul_precision(_default_float_32_precision)