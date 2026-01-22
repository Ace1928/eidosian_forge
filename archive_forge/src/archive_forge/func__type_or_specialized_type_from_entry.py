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
def _type_or_specialized_type_from_entry(self, entry):
    if entry and entry.is_type:
        if entry.type.is_fused and self.fused_to_specific:
            return entry.type.specialize(self.fused_to_specific)
        return entry.type