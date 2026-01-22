from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
class NonManglingModuleScope(Symtab.ModuleScope):

    def __init__(self, prefix, *args, **kw):
        self.prefix = prefix
        self.cython_scope = None
        self.cpp = kw.pop('cpp', False)
        Symtab.ModuleScope.__init__(self, *args, **kw)

    def add_imported_entry(self, name, entry, pos):
        entry.used = True
        return super(NonManglingModuleScope, self).add_imported_entry(name, entry, pos)

    def mangle(self, prefix, name=None):
        if name:
            if prefix in (Naming.typeobj_prefix, Naming.func_prefix, Naming.var_prefix, Naming.pyfunc_prefix):
                prefix = self.prefix
            return '%s%s' % (prefix, name)
        else:
            return Symtab.ModuleScope.mangle(self, prefix)