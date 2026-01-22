from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def _lift_binary_operands(left: typing.Any, right: typing.Any) -> tuple[Expr, Expr]:
    """Lift two binary operands simultaneously, inferring the widths of integer literals in either
    position to match the other operand."""
    left_int = isinstance(left, int) and (not isinstance(left, bool))
    right_int = isinstance(right, int) and (not isinstance(right, bool))
    if not (left_int or right_int):
        left = lift(left)
        right = lift(right)
    elif not right_int:
        right = lift(right)
        if right.type.kind is types.Uint:
            if left.bit_length() > right.type.width:
                raise TypeError(f"integer literal '{left}' is wider than the other operand '{right}'")
            left = Value(left, right.type)
        else:
            left = lift(left)
    elif not left_int:
        left = lift(left)
        if left.type.kind is types.Uint:
            if right.bit_length() > left.type.width:
                raise TypeError(f"integer literal '{right}' is wider than the other operand '{left}'")
            right = Value(right, left.type)
        else:
            right = lift(right)
    else:
        uint = types.Uint(max(left.bit_length(), right.bit_length(), 1))
        left = Value(left, uint)
        right = Value(right, uint)
    return (left, right)