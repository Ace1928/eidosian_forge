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
@optimizeconst
def visit_Concat(self, node: nodes.Concat, frame: Frame) -> None:
    if frame.eval_ctx.volatile:
        func_name = '(markup_join if context.eval_ctx.volatile else str_join)'
    elif frame.eval_ctx.autoescape:
        func_name = 'markup_join'
    else:
        func_name = 'str_join'
    self.write(f'{func_name}((')
    for arg in node.nodes:
        self.visit(arg, frame)
        self.write(', ')
    self.write('))')