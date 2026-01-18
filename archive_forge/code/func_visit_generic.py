from __future__ import annotations
import typing
from . import expr
def visit_generic(self, node: expr.Expr, /) -> _T_co:
    raise RuntimeError(f'expression visitor {self} has no method to handle expr {node}')