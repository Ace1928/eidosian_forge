import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class TemplateData(Literal):
    """A constant template string."""
    fields = ('data',)
    data: str

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> str:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        if eval_ctx.autoescape:
            return Markup(self.data)
        return self.data