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
def visit_FilterBlock(self, node: nodes.FilterBlock, frame: Frame) -> None:
    filter_frame = frame.inner()
    filter_frame.symbols.analyze_node(node)
    self.enter_frame(filter_frame)
    self.buffer(filter_frame)
    self.blockvisit(node.body, filter_frame)
    self.start_write(frame, node)
    self.visit_Filter(node.filter, filter_frame)
    self.end_write(frame)
    self.leave_frame(filter_frame)