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
def _output_child_pre(self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo) -> None:
    """Output extra source code before visiting a child of an
        ``Output`` node.
        """
    if frame.eval_ctx.volatile:
        self.write('(escape if context.eval_ctx.autoescape else str)(')
    elif frame.eval_ctx.autoescape:
        self.write('escape(')
    else:
        self.write('str(')
    if finalize.src is not None:
        self.write(finalize.src)