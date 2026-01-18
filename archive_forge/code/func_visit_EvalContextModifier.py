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
def visit_EvalContextModifier(self, node: nodes.EvalContextModifier, frame: Frame) -> None:
    for keyword in node.options:
        self.writeline(f'context.eval_ctx.{keyword.key} = ')
        self.visit(keyword.value, frame)
        try:
            val = keyword.value.as_const(frame.eval_ctx)
        except nodes.Impossible:
            frame.eval_ctx.volatile = True
        else:
            setattr(frame.eval_ctx, keyword.key, val)