import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_toplevel(self):
    """Traverse a template structure for module-level directives and
        generate the start of module-level code.

        """
    inherit = []
    namespaces = {}
    module_code = []
    self.compiler.pagetag = None

    class FindTopLevel:

        def visitInheritTag(s, node):
            inherit.append(node)

        def visitNamespaceTag(s, node):
            namespaces[node.name] = node

        def visitPageTag(s, node):
            self.compiler.pagetag = node

        def visitCode(s, node):
            if node.ismodule:
                module_code.append(node)
    f = FindTopLevel()
    for n in self.node.nodes:
        n.accept_visitor(f)
    self.compiler.namespaces = namespaces
    module_ident = set()
    for n in module_code:
        module_ident = module_ident.union(n.declared_identifiers())
    module_identifiers = _Identifiers(self.compiler)
    module_identifiers.declared = module_ident
    if self.compiler.generate_magic_comment and self.compiler.source_encoding:
        self.printer.writeline('# -*- coding:%s -*-' % self.compiler.source_encoding)
    if self.compiler.future_imports:
        self.printer.writeline('from __future__ import %s' % (', '.join(self.compiler.future_imports),))
    self.printer.writeline('from mako import runtime, filters, cache')
    self.printer.writeline('UNDEFINED = runtime.UNDEFINED')
    self.printer.writeline('STOP_RENDERING = runtime.STOP_RENDERING')
    self.printer.writeline('__M_dict_builtin = dict')
    self.printer.writeline('__M_locals_builtin = locals')
    self.printer.writeline('_magic_number = %r' % MAGIC_NUMBER)
    self.printer.writeline('_modified_time = %r' % time.time())
    self.printer.writeline('_enable_loop = %r' % self.compiler.enable_loop)
    self.printer.writeline('_template_filename = %r' % self.compiler.filename)
    self.printer.writeline('_template_uri = %r' % self.compiler.uri)
    self.printer.writeline('_source_encoding = %r' % self.compiler.source_encoding)
    if self.compiler.imports:
        buf = ''
        for imp in self.compiler.imports:
            buf += imp + '\n'
            self.printer.writeline(imp)
        impcode = ast.PythonCode(buf, source='', lineno=0, pos=0, filename='template defined imports')
    else:
        impcode = None
    main_identifiers = module_identifiers.branch(self.node)
    mit = module_identifiers.topleveldefs
    module_identifiers.topleveldefs = mit.union(main_identifiers.topleveldefs)
    module_identifiers.declared.update(TOPLEVEL_DECLARED)
    if impcode:
        module_identifiers.declared.update(impcode.declared_identifiers)
    self.compiler.identifiers = module_identifiers
    self.printer.writeline('_exports = %r' % [n.name for n in main_identifiers.topleveldefs.values()])
    self.printer.write_blanks(2)
    if len(module_code):
        self.write_module_code(module_code)
    if len(inherit):
        self.write_namespaces(namespaces)
        self.write_inherit(inherit[-1])
    elif len(namespaces):
        self.write_namespaces(namespaces)
    return list(main_identifiers.topleveldefs.values())