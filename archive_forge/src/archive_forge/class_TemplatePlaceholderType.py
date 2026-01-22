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
class TemplatePlaceholderType(CType):

    def __init__(self, name, optional=False):
        self.name = name
        self.optional = optional

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if entity_code:
            return self.name + ' ' + entity_code
        else:
            return self.name

    def specialize(self, values):
        if self in values:
            return values[self]
        else:
            return self

    def deduce_template_params(self, actual):
        return {self: actual}

    def same_as_resolved_type(self, other_type):
        if isinstance(other_type, TemplatePlaceholderType):
            return self.name == other_type.name
        else:
            return 0

    def __hash__(self):
        return hash(self.name)

    def __cmp__(self, other):
        if isinstance(other, TemplatePlaceholderType):
            return cmp(self.name, other.name)
        else:
            return cmp(type(self), type(other))

    def __eq__(self, other):
        if isinstance(other, TemplatePlaceholderType):
            return self.name == other.name
        else:
            return False