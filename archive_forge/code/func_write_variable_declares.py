import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_variable_declares(self, identifiers, toplevel=False, limit=None):
    """write variable declarations at the top of a function.

        the variable declarations are in the form of callable
        definitions for defs and/or name lookup within the
        function's context argument. the names declared are based
        on the names that are referenced in the function body,
        which don't otherwise have any explicit assignment
        operation. names that are assigned within the body are
        assumed to be locally-scoped variables and are not
        separately declared.

        for def callable definitions, if the def is a top-level
        callable then a 'stub' callable is generated which wraps
        the current Context into a closure. if the def is not
        top-level, it is fully rendered as a local closure.

        """
    comp_idents = {c.funcname: c for c in identifiers.defs}
    to_write = set()
    to_write = to_write.union(identifiers.undeclared)
    to_write = to_write.union([c.funcname for c in identifiers.closuredefs.values()])
    to_write = to_write.difference(identifiers.argument_declared)
    to_write = to_write.difference(identifiers.locally_declared)
    if self.compiler.enable_loop:
        has_loop = 'loop' in to_write
        to_write.discard('loop')
    else:
        has_loop = False
    if limit is not None:
        to_write = to_write.intersection(limit)
    if toplevel and getattr(self.compiler, 'has_ns_imports', False):
        self.printer.writeline('_import_ns = {}')
        self.compiler.has_imports = True
        for ident, ns in self.compiler.namespaces.items():
            if 'import' in ns.attributes:
                self.printer.writeline('_mako_get_namespace(context, %r)._populate(_import_ns, %r)' % (ident, re.split('\\s*,\\s*', ns.attributes['import'])))
    if has_loop:
        self.printer.writeline('loop = __M_loop = runtime.LoopStack()')
    for ident in to_write:
        if ident in comp_idents:
            comp = comp_idents[ident]
            if comp.is_block:
                if not comp.is_anonymous:
                    self.write_def_decl(comp, identifiers)
                else:
                    self.write_inline_def(comp, identifiers, nested=True)
            elif comp.is_root():
                self.write_def_decl(comp, identifiers)
            else:
                self.write_inline_def(comp, identifiers, nested=True)
        elif ident in self.compiler.namespaces:
            self.printer.writeline('%s = _mako_get_namespace(context, %r)' % (ident, ident))
        elif getattr(self.compiler, 'has_ns_imports', False):
            if self.compiler.strict_undefined:
                self.printer.writelines('%s = _import_ns.get(%r, UNDEFINED)' % (ident, ident), 'if %s is UNDEFINED:' % ident, 'try:', '%s = context[%r]' % (ident, ident), 'except KeyError:', 'raise NameError("\'%s\' is not defined")' % ident, None, None)
            else:
                self.printer.writeline('%s = _import_ns.get(%r, context.get(%r, UNDEFINED))' % (ident, ident, ident))
        elif self.compiler.strict_undefined:
            self.printer.writelines('try:', '%s = context[%r]' % (ident, ident), 'except KeyError:', 'raise NameError("\'%s\' is not defined")' % ident, None)
        else:
            self.printer.writeline('%s = context.get(%r, UNDEFINED)' % (ident, ident))
    self.printer.writeline('__M_writer = context.writer()')