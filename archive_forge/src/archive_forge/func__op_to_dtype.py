from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
def _op_to_dtype(self, a, dtype, src_dtype=None):
    assert dtype in ('torch.float32', 'torch.float16'), f'Unsupported dtype: {dtype}'
    assert src_dtype in (None, 'torch.float32', 'torch.float16'), f'Unsupported source dtype: {src_dtype}'
    return a