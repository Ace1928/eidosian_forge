import functools
import sys
import typing as t
from collections import abc
from itertools import chain
from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context
class BlockReference:
    """One block on a template reference."""

    def __init__(self, name: str, context: 'Context', stack: t.List[t.Callable[['Context'], t.Iterator[str]]], depth: int) -> None:
        self.name = name
        self._context = context
        self._stack = stack
        self._depth = depth

    @property
    def super(self) -> t.Union['BlockReference', 'Undefined']:
        """Super the block."""
        if self._depth + 1 >= len(self._stack):
            return self._context.environment.undefined(f'there is no parent block called {self.name!r}.', name='super')
        return BlockReference(self.name, self._context, self._stack, self._depth + 1)

    @internalcode
    async def _async_call(self) -> str:
        rv = concat([x async for x in self._stack[self._depth](self._context)])
        if self._context.eval_ctx.autoescape:
            return Markup(rv)
        return rv

    @internalcode
    def __call__(self) -> str:
        if self._context.environment.is_async:
            return self._async_call()
        rv = concat(self._stack[self._depth](self._context))
        if self._context.eval_ctx.autoescape:
            return Markup(rv)
        return rv