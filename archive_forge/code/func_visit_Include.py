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
def visit_Include(self, node: nodes.Include, frame: Frame) -> None:
    """Handles includes."""
    if node.ignore_missing:
        self.writeline('try:')
        self.indent()
    func_name = 'get_or_select_template'
    if isinstance(node.template, nodes.Const):
        if isinstance(node.template.value, str):
            func_name = 'get_template'
        elif isinstance(node.template.value, (tuple, list)):
            func_name = 'select_template'
    elif isinstance(node.template, (nodes.Tuple, nodes.List)):
        func_name = 'select_template'
    self.writeline(f'template = environment.{func_name}(', node)
    self.visit(node.template, frame)
    self.write(f', {self.name!r})')
    if node.ignore_missing:
        self.outdent()
        self.writeline('except TemplateNotFound:')
        self.indent()
        self.writeline('pass')
        self.outdent()
        self.writeline('else:')
        self.indent()
    skip_event_yield = False
    if node.with_context:
        self.writeline(f'{self.choose_async()}for event in template.root_render_func(template.new_context(context.get_all(), True, {self.dump_local_context(frame)})):')
    elif self.environment.is_async:
        self.writeline('for event in (await template._get_default_module_async())._body_stream:')
    else:
        self.writeline('yield from template._get_default_module()._body_stream')
        skip_event_yield = True
    if not skip_event_yield:
        self.indent()
        self.simple_write('event', frame)
        self.outdent()
    if node.ignore_missing:
        self.outdent()