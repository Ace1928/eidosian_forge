from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def analyse_sharing_attributes(self, env):
    """
        Analyse the privates for this block and set them in self.privates.
        This should be called in a post-order fashion during the
        analyse_expressions phase
        """
    for entry, (pos, op) in self.assignments.items():
        if self.is_prange and (not self.is_parallel):
            if entry in self.parent.assignments:
                error(pos, 'Cannot assign to private of outer parallel block')
                continue
        if not self.is_prange and op:
            error(pos, 'Reductions not allowed for parallel blocks')
            continue
        lastprivate = True
        self.propagate_var_privatization(entry, pos, op, lastprivate)