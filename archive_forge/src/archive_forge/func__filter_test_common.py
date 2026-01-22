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
@contextmanager
def _filter_test_common(self, node: t.Union[nodes.Filter, nodes.Test], frame: Frame, is_filter: bool) -> t.Iterator[None]:
    if self.environment.is_async:
        self.write('(await auto_await(')
    if is_filter:
        self.write(f'{self.filters[node.name]}(')
        func = self.environment.filters.get(node.name)
    else:
        self.write(f'{self.tests[node.name]}(')
        func = self.environment.tests.get(node.name)
    if func is None and (not frame.soft_frame):
        type_name = 'filter' if is_filter else 'test'
        self.fail(f'No {type_name} named {node.name!r}.', node.lineno)
    pass_arg = {_PassArg.context: 'context', _PassArg.eval_context: 'context.eval_ctx', _PassArg.environment: 'environment'}.get(_PassArg.from_obj(func))
    if pass_arg is not None:
        self.write(f'{pass_arg}, ')
    yield
    self.signature(node, frame)
    self.write(')')
    if self.environment.is_async:
        self.write('))')