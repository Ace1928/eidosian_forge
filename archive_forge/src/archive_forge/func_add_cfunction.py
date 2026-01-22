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
def add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=False):
    prev_entry = self.lookup_here(name)
    entry = ClassScope.add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=inherited)
    entry.is_cmethod = 1
    entry.prev_entry = prev_entry
    return entry