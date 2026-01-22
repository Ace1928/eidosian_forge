import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class _FilterTestCommon(Expr):
    fields = ('node', 'name', 'args', 'kwargs', 'dyn_args', 'dyn_kwargs')
    node: Expr
    name: str
    args: t.List[Expr]
    kwargs: t.List[Pair]
    dyn_args: t.Optional[Expr]
    dyn_kwargs: t.Optional[Expr]
    abstract = True
    _is_filter = True

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        if self._is_filter:
            env_map = eval_ctx.environment.filters
        else:
            env_map = eval_ctx.environment.tests
        func = env_map.get(self.name)
        pass_arg = _PassArg.from_obj(func)
        if func is None or pass_arg is _PassArg.context:
            raise Impossible()
        if eval_ctx.environment.is_async and (getattr(func, 'jinja_async_variant', False) is True or inspect.iscoroutinefunction(func)):
            raise Impossible()
        args, kwargs = args_as_const(self, eval_ctx)
        args.insert(0, self.node.as_const(eval_ctx))
        if pass_arg is _PassArg.eval_context:
            args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            args.insert(0, eval_ctx.environment)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise Impossible() from e