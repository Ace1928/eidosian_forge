import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class CondExpr(Expr):
    """A conditional expression (inline if expression).  (``{{
    foo if bar else baz }}``)
    """
    fields = ('test', 'expr1', 'expr2')
    test: Expr
    expr1: Expr
    expr2: t.Optional[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if self.test.as_const(eval_ctx):
            return self.expr1.as_const(eval_ctx)
        if self.expr2 is None:
            raise Impossible()
        return self.expr2.as_const(eval_ctx)