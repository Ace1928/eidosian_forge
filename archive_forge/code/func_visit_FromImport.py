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
def visit_FromImport(self, node: nodes.FromImport, frame: Frame) -> None:
    """Visit named imports."""
    self.newline(node)
    self.write('included_template = ')
    self._import_common(node, frame)
    var_names = []
    discarded_names = []
    for name in node.names:
        if isinstance(name, tuple):
            name, alias = name
        else:
            alias = name
        self.writeline(f'{frame.symbols.ref(alias)} = getattr(included_template, {name!r}, missing)')
        self.writeline(f'if {frame.symbols.ref(alias)} is missing:')
        self.indent()
        message = f'the template {{included_template.__name__!r}} (imported on {self.position(node)}) does not export the requested name {name!r}'
        self.writeline(f'{frame.symbols.ref(alias)} = undefined(f{message!r}, name={name!r})')
        self.outdent()
        if frame.toplevel:
            var_names.append(alias)
            if not alias.startswith('_'):
                discarded_names.append(alias)
    if var_names:
        if len(var_names) == 1:
            name = var_names[0]
            self.writeline(f'context.vars[{name!r}] = {frame.symbols.ref(name)}')
        else:
            names_kv = ', '.join((f'{name!r}: {frame.symbols.ref(name)}' for name in var_names))
            self.writeline(f'context.vars.update({{{names_kv}}})')
    if discarded_names:
        if len(discarded_names) == 1:
            self.writeline(f'context.exported_vars.discard({discarded_names[0]!r})')
        else:
            names_str = ', '.join(map(repr, discarded_names))
            self.writeline(f'context.exported_vars.difference_update(({names_str}))')