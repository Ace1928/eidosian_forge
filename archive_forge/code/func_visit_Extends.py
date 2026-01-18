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
def visit_Extends(self, node: nodes.Extends, frame: Frame) -> None:
    """Calls the extender."""
    if not frame.toplevel:
        self.fail('cannot use extend from a non top-level scope', node.lineno)
    if self.extends_so_far > 0:
        if not self.has_known_extends:
            self.writeline('if parent_template is not None:')
            self.indent()
        self.writeline('raise TemplateRuntimeError("extended multiple times")')
        if self.has_known_extends:
            raise CompilerExit()
        else:
            self.outdent()
    self.writeline('parent_template = environment.get_template(', node)
    self.visit(node.template, frame)
    self.write(f', {self.name!r})')
    self.writeline('for name, parent_block in parent_template.blocks.items():')
    self.indent()
    self.writeline('context.blocks.setdefault(name, []).append(parent_block)')
    self.outdent()
    if frame.rootlevel:
        self.has_known_extends = True
    self.extends_so_far += 1