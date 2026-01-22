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
class CConstOrVolatileScope(Scope):

    def __init__(self, base_type_scope, is_const=0, is_volatile=0):
        Scope.__init__(self, 'cv_' + base_type_scope.name, base_type_scope.outer_scope, base_type_scope.parent_scope)
        self.base_type_scope = base_type_scope
        self.is_const = is_const
        self.is_volatile = is_volatile

    def lookup_here(self, name):
        entry = self.base_type_scope.lookup_here(name)
        if entry is not None:
            entry = copy.copy(entry)
            entry.type = PyrexTypes.c_const_or_volatile_type(entry.type, self.is_const, self.is_volatile)
            return entry