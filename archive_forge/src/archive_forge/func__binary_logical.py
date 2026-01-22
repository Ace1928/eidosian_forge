from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def _binary_logical(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    bool_ = types.Bool()
    left = _coerce_lossless(lift(left), bool_)
    right = _coerce_lossless(lift(right), bool_)
    return Binary(op, left, right, bool_)