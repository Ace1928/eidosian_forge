from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CConstOrVolatileType(BaseType):
    """A C const or volatile type"""
    subtypes = ['cv_base_type']
    is_cv_qualified = 1

    def __init__(self, base_type, is_const=0, is_volatile=0):
        self.cv_base_type = base_type
        self.is_const = is_const
        self.is_volatile = is_volatile
        if base_type.has_attributes and base_type.scope is not None:
            from .Symtab import CConstOrVolatileScope
            self.scope = CConstOrVolatileScope(base_type.scope, is_const, is_volatile)

    def cv_string(self):
        cvstring = ''
        if self.is_const:
            cvstring = 'const ' + cvstring
        if self.is_volatile:
            cvstring = 'volatile ' + cvstring
        return cvstring

    def __repr__(self):
        return '<CConstOrVolatileType %s%r>' % (self.cv_string(), self.cv_base_type)

    def __str__(self):
        return self.declaration_code('', for_display=1)

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        cv = self.cv_string()
        if for_display or pyrex:
            return cv + self.cv_base_type.declaration_code(entity_code, for_display, dll_linkage, pyrex)
        else:
            return self.cv_base_type.declaration_code(cv + entity_code, for_display, dll_linkage, pyrex)

    def specialize(self, values):
        base_type = self.cv_base_type.specialize(values)
        if base_type == self.cv_base_type:
            return self
        return CConstOrVolatileType(base_type, self.is_const, self.is_volatile)

    def deduce_template_params(self, actual):
        return self.cv_base_type.deduce_template_params(actual)

    def can_coerce_to_pyobject(self, env):
        return self.cv_base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.cv_base_type.can_coerce_from_pyobject(env)

    def create_to_py_utility_code(self, env):
        if self.cv_base_type.create_to_py_utility_code(env):
            self.to_py_function = self.cv_base_type.to_py_function
            return True

    def same_as_resolved_type(self, other_type):
        if other_type.is_cv_qualified:
            return self.cv_base_type.same_as_resolved_type(other_type.cv_base_type)
        return self.cv_base_type.same_as_resolved_type(other_type)

    def __getattr__(self, name):
        return getattr(self.cv_base_type, name)