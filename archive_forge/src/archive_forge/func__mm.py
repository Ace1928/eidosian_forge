import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def _mm(self, B: torch.Tensor, *, prefer_col_major_output: bool=False, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
    if isinstance(B, Sparse24Tensor):
        raise ValueError('`Sparse24Tensor @ Sparse24Tensor` is not supported by the hardware')
    if self.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(f'`{self.__class__.__name__}` matmul: Broadcasting is not implemented')
    if self.shape[1] != B.shape[0]:
        raise NotImplementedError(f'`{self.__class__.__name__}` matmul: invalid shapes     ({self.shape[0]}, {self.shape[1]}) @ ({B.shape[0]}, {B.shape[1]})')
    if B.shape[1] % 8 != 0:
        raise NotImplementedError(f'`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`. The dense matrix B should have the second dimension aligned to 8.')
    if B.dtype != self.dtype:
        raise NotImplementedError(f'`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, with A.dtype={self.dtype} and B.dtype={B.dtype}. This operation is only supported when A and B have the same data type.')
    if bias is not None and bias.dtype != self.dtype:
        raise NotImplementedError(f'`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, with A.dtype=B.dtype={{self.dtype}} and C.dtype={{B.dtype}}. This operation is only supported when A, B and C have the same data type.')
    assert _has_cusparseLt()
    out = Sp24GemmCusplt.OPERATOR(self.packed, B, bias=bias, transpose_result=prefer_col_major_output)
    if prefer_col_major_output:
        out = out.t()
    return out[:self.shape[0]]