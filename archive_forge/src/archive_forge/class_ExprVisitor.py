from __future__ import annotations
import typing
from . import expr
class ExprVisitor(typing.Generic[_T_co]):
    """Base class for visitors to the :class:`Expr` tree.  Subclasses should override whichever of
    the ``visit_*`` methods that they are able to handle, and should be organised such that
    non-existent methods will never be called."""
    __slots__ = ()

    def visit_generic(self, node: expr.Expr, /) -> _T_co:
        raise RuntimeError(f'expression visitor {self} has no method to handle expr {node}')

    def visit_var(self, node: expr.Var, /) -> _T_co:
        return self.visit_generic(node)

    def visit_value(self, node: expr.Value, /) -> _T_co:
        return self.visit_generic(node)

    def visit_unary(self, node: expr.Unary, /) -> _T_co:
        return self.visit_generic(node)

    def visit_binary(self, node: expr.Binary, /) -> _T_co:
        return self.visit_generic(node)

    def visit_cast(self, node: expr.Cast, /) -> _T_co:
        return self.visit_generic(node)