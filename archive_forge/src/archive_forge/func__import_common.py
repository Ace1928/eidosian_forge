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
def _import_common(self, node: t.Union[nodes.Import, nodes.FromImport], frame: Frame) -> None:
    self.write(f'{self.choose_async('await ')}environment.get_template(')
    self.visit(node.template, frame)
    self.write(f', {self.name!r}).')
    if node.with_context:
        f_name = f'make_module{self.choose_async('_async')}'
        self.write(f'{f_name}(context.get_all(), True, {self.dump_local_context(frame)})')
    else:
        self.write(f'_get_default_module{self.choose_async('_async')}(context)')