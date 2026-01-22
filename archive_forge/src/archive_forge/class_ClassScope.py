from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class ClassScope(Scope):
    scope_predefined_names = ['__module__', '__qualname__']

    def mangle_class_private_name(self, name):
        if name and name.lower().startswith('__pyx_'):
            return name
        if name and name.startswith('__') and (not name.endswith('__')):
            name = EncodedString('_%s%s' % (self.class_name.lstrip('_'), name))
        return name

    def __init__(self, name, outer_scope):
        Scope.__init__(self, name, outer_scope, outer_scope)
        self.class_name = name
        self.doc = None

    def lookup(self, name):
        entry = Scope.lookup(self, name)
        if entry:
            return entry
        if name == 'classmethod':
            entry = Entry('classmethod', '__Pyx_Method_ClassMethod', PyrexTypes.CFuncType(py_object_type, [PyrexTypes.CFuncTypeArg('', py_object_type, None)], 0, 0))
            entry.utility_code_definition = Code.UtilityCode.load_cached('ClassMethod', 'CythonFunction.c')
            self.use_entry_utility_code(entry)
            entry.is_cfunction = 1
        return entry