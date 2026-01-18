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
def visit_Template(self, node: nodes.Template, frame: t.Optional[Frame]=None) -> None:
    assert frame is None, 'no root frame allowed'
    eval_ctx = EvalContext(self.environment, self.name)
    from .runtime import exported, async_exported
    if self.environment.is_async:
        exported_names = sorted(exported + async_exported)
    else:
        exported_names = sorted(exported)
    self.writeline('from jinja2.runtime import ' + ', '.join(exported_names))
    envenv = '' if self.defer_init else ', environment=environment'
    have_extends = node.find(nodes.Extends) is not None
    for block in node.find_all(nodes.Block):
        if block.name in self.blocks:
            self.fail(f'block {block.name!r} defined twice', block.lineno)
        self.blocks[block.name] = block
    for import_ in node.find_all(nodes.ImportedName):
        if import_.importname not in self.import_aliases:
            imp = import_.importname
            self.import_aliases[imp] = alias = self.temporary_identifier()
            if '.' in imp:
                module, obj = imp.rsplit('.', 1)
                self.writeline(f'from {module} import {obj} as {alias}')
            else:
                self.writeline(f'import {imp} as {alias}')
    self.writeline(f'name = {self.name!r}')
    self.writeline(f'{self.func('root')}(context, missing=missing{envenv}):', extra=1)
    self.indent()
    self.write_commons()
    frame = Frame(eval_ctx)
    if 'self' in find_undeclared(node.body, ('self',)):
        ref = frame.symbols.declare_parameter('self')
        self.writeline(f'{ref} = TemplateReference(context)')
    frame.symbols.analyze_node(node)
    frame.toplevel = frame.rootlevel = True
    frame.require_output_check = have_extends and (not self.has_known_extends)
    if have_extends:
        self.writeline('parent_template = None')
    self.enter_frame(frame)
    self.pull_dependencies(node.body)
    self.blockvisit(node.body, frame)
    self.leave_frame(frame, with_python_scope=True)
    self.outdent()
    if have_extends:
        if not self.has_known_extends:
            self.indent()
            self.writeline('if parent_template is not None:')
        self.indent()
        if not self.environment.is_async:
            self.writeline('yield from parent_template.root_render_func(context)')
        else:
            self.writeline('async for event in parent_template.root_render_func(context):')
            self.indent()
            self.writeline('yield event')
            self.outdent()
        self.outdent(1 + (not self.has_known_extends))
    for name, block in self.blocks.items():
        self.writeline(f'{self.func('block_' + name)}(context, missing=missing{envenv}):', block, 1)
        self.indent()
        self.write_commons()
        block_frame = Frame(eval_ctx)
        block_frame.block_frame = True
        undeclared = find_undeclared(block.body, ('self', 'super'))
        if 'self' in undeclared:
            ref = block_frame.symbols.declare_parameter('self')
            self.writeline(f'{ref} = TemplateReference(context)')
        if 'super' in undeclared:
            ref = block_frame.symbols.declare_parameter('super')
            self.writeline(f'{ref} = context.super({name!r}, block_{name})')
        block_frame.symbols.analyze_node(block)
        block_frame.block = name
        self.writeline('_block_vars = {}')
        self.enter_frame(block_frame)
        self.pull_dependencies(block.body)
        self.blockvisit(block.body, block_frame)
        self.leave_frame(block_frame, with_python_scope=True)
        self.outdent()
    blocks_kv_str = ', '.join((f'{x!r}: block_{x}' for x in self.blocks))
    self.writeline(f'blocks = {{{blocks_kv_str}}}', extra=1)
    debug_kv_str = '&'.join((f'{k}={v}' for k, v in self.debug_info))
    self.writeline(f'debug_info = {debug_kv_str!r}')