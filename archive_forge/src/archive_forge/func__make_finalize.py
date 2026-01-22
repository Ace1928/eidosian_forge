import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def _make_finalize(self) -> _FinalizeInfo:
    """Build the finalize function to be used on constants and at
        runtime. Cached so it's only created once for all output nodes.

        Returns a ``namedtuple`` with the following attributes:

        ``const``
            A function to finalize constant data at compile time.

        ``src``
            Source code to output around nodes to be evaluated at
            runtime.
        """
    if self._finalize is not None:
        return self._finalize
    finalize: t.Optional[t.Callable[..., t.Any]]
    finalize = default = self._default_finalize
    src = None
    if self.environment.finalize:
        src = 'environment.finalize('
        env_finalize = self.environment.finalize
        pass_arg = {_PassArg.context: 'context', _PassArg.eval_context: 'context.eval_ctx', _PassArg.environment: 'environment'}.get(_PassArg.from_obj(env_finalize))
        finalize = None
        if pass_arg is None:

            def finalize(value: t.Any) -> t.Any:
                return default(env_finalize(value))
        else:
            src = f'{src}{pass_arg}, '
            if pass_arg == 'environment':

                def finalize(value: t.Any) -> t.Any:
                    return default(env_finalize(self.environment, value))
    self._finalize = self._FinalizeInfo(finalize, src)
    return self._finalize