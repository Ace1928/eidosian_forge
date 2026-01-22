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
def add_imported_entry(self, name, entry, pos):
    if entry.is_pyglobal:
        entry.is_variable = True
    if entry not in self.entries:
        self.entries[name] = entry
    else:
        warning(pos, "'%s' redeclared  " % name, 0)