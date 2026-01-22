from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _op_relu(self, a):
    const_zero = self._op_constant(0.0, 'torch.float32')
    return '{' + str(a) + ', ' + const_zero + '}'