from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty
        if not shape:
            raise TypeError('0d block_type is forbidden')
        if isinstance(shape[0], constexpr):
            shape = [s.value for s in shape]
        self.shape = shape
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        if self.numel > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError(f'numel ({self.numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})')
        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return f'<{self.shape}, {self.element_ty}>'

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> List[int]:
        return self.shape

    def __eq__(self, other: block_type) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: block_type) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self.element_ty