from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
class DataArrayGroupByOpsMixin:
    __slots__ = ()

    def _binary_op(self, other: T_Xarray, f: Callable, reflexive: bool=False) -> T_Xarray:
        raise NotImplementedError

    def __add__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.add)

    def __sub__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.sub)

    def __mul__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.mul)

    def __pow__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.pow)

    def __truediv__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.truediv)

    def __floordiv__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.floordiv)

    def __mod__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.mod)

    def __and__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.and_)

    def __xor__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.xor)

    def __or__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.or_)

    def __lshift__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.lshift)

    def __rshift__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.rshift)

    def __lt__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.lt)

    def __le__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.le)

    def __gt__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.gt)

    def __ge__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.ge)

    def __eq__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, nputils.array_eq)

    def __ne__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, nputils.array_ne)
    __hash__: None

    def __radd__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.add, reflexive=True)

    def __rsub__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.sub, reflexive=True)

    def __rmul__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.mul, reflexive=True)

    def __rpow__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.pow, reflexive=True)

    def __rtruediv__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.truediv, reflexive=True)

    def __rfloordiv__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.floordiv, reflexive=True)

    def __rmod__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.mod, reflexive=True)

    def __rand__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.and_, reflexive=True)

    def __rxor__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.xor, reflexive=True)

    def __ror__(self, other: T_Xarray) -> T_Xarray:
        return self._binary_op(other, operator.or_, reflexive=True)
    __add__.__doc__ = operator.add.__doc__
    __sub__.__doc__ = operator.sub.__doc__
    __mul__.__doc__ = operator.mul.__doc__
    __pow__.__doc__ = operator.pow.__doc__
    __truediv__.__doc__ = operator.truediv.__doc__
    __floordiv__.__doc__ = operator.floordiv.__doc__
    __mod__.__doc__ = operator.mod.__doc__
    __and__.__doc__ = operator.and_.__doc__
    __xor__.__doc__ = operator.xor.__doc__
    __or__.__doc__ = operator.or_.__doc__
    __lshift__.__doc__ = operator.lshift.__doc__
    __rshift__.__doc__ = operator.rshift.__doc__
    __lt__.__doc__ = operator.lt.__doc__
    __le__.__doc__ = operator.le.__doc__
    __gt__.__doc__ = operator.gt.__doc__
    __ge__.__doc__ = operator.ge.__doc__
    __eq__.__doc__ = nputils.array_eq.__doc__
    __ne__.__doc__ = nputils.array_ne.__doc__
    __radd__.__doc__ = operator.add.__doc__
    __rsub__.__doc__ = operator.sub.__doc__
    __rmul__.__doc__ = operator.mul.__doc__
    __rpow__.__doc__ = operator.pow.__doc__
    __rtruediv__.__doc__ = operator.truediv.__doc__
    __rfloordiv__.__doc__ = operator.floordiv.__doc__
    __rmod__.__doc__ = operator.mod.__doc__
    __rand__.__doc__ = operator.and_.__doc__
    __rxor__.__doc__ = operator.xor.__doc__
    __ror__.__doc__ = operator.or_.__doc__