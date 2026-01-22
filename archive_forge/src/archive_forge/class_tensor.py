from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
class tensor:

    def __init__(self, handle, type: dtype):
        self.handle = handle
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type
        self.dtype = type.scalar
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:
        return str(self.dtype) + '[' + ', '.join((str(s) for s in self.shape)) + ']'

    @builtin
    def __add__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.add(self, other, _builder)

    @builtin
    def __radd__(self, other, _builder=None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(self, other, _builder)

    @builtin
    def __rsub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mul(self, other, _builder)

    @builtin
    def __rmul__(self, other, _builder=None):
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(self, other, _builder)

    @builtin
    def __rtruediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.floordiv(self, other, _builder)

    @builtin
    def __rfloordiv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.floordiv(other, self, _builder)

    @builtin
    def __mod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(other, self, _builder)

    @builtin
    def __neg__(self, _builder=None):
        return semantic.minus(self, _builder)

    @builtin
    def __invert__(self, _builder=None):
        return semantic.invert(self, _builder)

    @builtin
    def __and__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.and_(self, other, _builder)

    @builtin
    def __rand__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.and_(other, self, _builder)

    @builtin
    def __or__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.or_(self, other, _builder)

    @builtin
    def __ror__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.or_(other, self, _builder)

    @builtin
    def __xor__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.xor_(self, other, _builder)

    @builtin
    def __rxor__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.xor_(other, self, _builder)

    @builtin
    def __lshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.shl(self, other, _builder)

    @builtin
    def __rlshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.shl(other, self, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        if self.dtype.is_int_signed():
            return semantic.ashr(self, other, _builder)
        else:
            return semantic.lshr(self, other, _builder)

    @builtin
    def __rrshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        if self.dtype.is_int_signed():
            return semantic.ashr(other, self, _builder)
        else:
            return semantic.lshr(other, self, _builder)

    @builtin
    def __gt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(other, self, _builder)

    @builtin
    def __ge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(other, self, _builder)

    @builtin
    def __lt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(other, self, _builder)

    @builtin
    def __le__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(other, self, _builder)

    @builtin
    def __eq__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.equal(self, other, _builder)

    @builtin
    def __req__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.equal(other, self, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(self, other, _builder)

    @builtin
    def __rne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(other, self, _builder)

    @builtin
    def logical_and(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.logical_and(self, other, _builder)

    @builtin
    def logical_or(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.logical_or(self, other, _builder)

    @builtin
    def __not__(self, _builder=None):
        return semantic.not_(self, _builder)

    @builtin
    def __getitem__(self, slices, _builder=None):
        if isinstance(slices, (slice, constexpr)):
            slices = [slices]
        ret = self
        for dim, sl in enumerate(slices):
            if sl is None or (isinstance(sl, constexpr) and sl.value is None):
                ret = semantic.expand_dims(ret, dim, _builder)
            elif isinstance(sl, slice) and sl.start is None and (sl.stop is None) and (sl.step is None):
                pass
            else:
                assert False, f'unsupported tensor index: {sl}'
        return ret

    @property
    def T(self):
        assert False, 'Transposition must be created by the AST Visitor'

    @builtin
    def to(self, dtype, bitcast=False, _builder=None):
        if isinstance(bitcast, constexpr):
            bitcast = bitcast.value
        if bitcast:
            return semantic.bitcast(self, dtype, _builder)
        return semantic.cast(self, dtype, _builder)