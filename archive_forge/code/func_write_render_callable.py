import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_render_callable(self, node, name, args, buffered, filtered, cached):
    """write a top-level render callable.

        this could be the main render() method or that of a top-level def."""
    if self.in_def:
        decorator = node.decorator
        if decorator:
            self.printer.writeline('@runtime._decorate_toplevel(%s)' % decorator)
    self.printer.start_source(node.lineno)
    self.printer.writelines('def %s(%s):' % (name, ','.join(args)), '__M_caller = context.caller_stack._push_frame()', 'try:')
    if buffered or filtered or cached:
        self.printer.writeline('context._push_buffer()')
    self.identifier_stack.append(self.compiler.identifiers.branch(self.node))
    if (not self.in_def or self.node.is_block) and '**pageargs' in args:
        self.identifier_stack[-1].argument_declared.add('pageargs')
    if not self.in_def and (len(self.identifiers.locally_assigned) > 0 or len(self.identifiers.argument_declared) > 0):
        self.printer.writeline('__M_locals = __M_dict_builtin(%s)' % ','.join(['%s=%s' % (x, x) for x in self.identifiers.argument_declared]))
    self.write_variable_declares(self.identifiers, toplevel=True)
    for n in self.node.nodes:
        n.accept_visitor(self)
    self.write_def_finish(self.node, buffered, filtered, cached)
    self.printer.writeline(None)
    self.printer.write_blanks(2)
    if cached:
        self.write_cache_decorator(node, name, args, buffered, self.identifiers, toplevel=True)