import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class Getitem(Expr):
    """Get an attribute or item from an expression and prefer the item."""
    fields = ('node', 'arg', 'ctx')
    node: Expr
    arg: Expr
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Any:
        if self.ctx != 'load':
            raise Impossible()
        eval_ctx = get_eval_context(self, eval_ctx)
        try:
            return eval_ctx.environment.getitem(self.node.as_const(eval_ctx), self.arg.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e