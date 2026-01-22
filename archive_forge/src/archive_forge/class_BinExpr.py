import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class BinExpr(Expr):
    """Baseclass for all binary expressions."""
    fields = ('left', 'right')
    left: Expr
    right: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.environment.sandboxed and self.operator in eval_ctx.environment.intercepted_binops:
            raise Impossible()
        f = _binop_to_func[self.operator]
        try:
            return f(self.left.as_const(eval_ctx), self.right.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e