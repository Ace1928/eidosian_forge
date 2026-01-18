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
def visit_Getitem(self, node: nodes.Getitem, frame: Frame) -> None:
    if isinstance(node.arg, nodes.Slice):
        self.visit(node.node, frame)
        self.write('[')
        self.visit(node.arg, frame)
        self.write(']')
    else:
        if self.environment.is_async:
            self.write('(await auto_await(')
        self.write('environment.getitem(')
        self.visit(node.node, frame)
        self.write(', ')
        self.visit(node.arg, frame)
        self.write(')')
        if self.environment.is_async:
            self.write('))')