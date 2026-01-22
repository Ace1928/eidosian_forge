import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Union[Markup, t.Any]:
    eval_ctx = get_eval_context(self, eval_ctx)
    if eval_ctx.volatile:
        raise Impossible()
    expr = self.expr.as_const(eval_ctx)
    if eval_ctx.autoescape:
        return Markup(expr)
    return expr