import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
class TritonKernelWrapperFunctional(HigherOrderOperator):

    def __init__(self):
        super().__init__('triton_kernel_wrapper_functional')