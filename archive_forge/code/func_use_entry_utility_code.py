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
def use_entry_utility_code(self, entry):
    if entry is None:
        return
    if entry.utility_code:
        self.utility_code_list.append(entry.utility_code)
    if entry.utility_code_definition:
        self.utility_code_list.append(entry.utility_code_definition)