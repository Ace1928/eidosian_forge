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
def _declare_pyfunction(self, name, pos, visibility='extern', entry=None):
    if entry and (not entry.type.is_cfunction):
        error(pos, "'%s' already declared" % name)
        error(entry.pos, 'Previous declaration is here')
    entry = self.declare_var(name, py_object_type, pos, visibility=visibility)
    entry.signature = pyfunction_signature
    self.pyfunc_entries.append(entry)
    return entry