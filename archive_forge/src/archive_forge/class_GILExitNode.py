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
class GILExitNode(StatNode):
    """
    Used as the 'finally' block in a GILStatNode

    state   string   'gil' or 'nogil'
    #   scope_gil_state_known  bool  For nogil functions this can be False, since they can also be run with gil
    #                           set to False by GilCheck transform
    """
    child_attrs = []
    state_temp = None
    scope_gil_state_known = True

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.state_temp:
            variable = self.state_temp.result()
        else:
            variable = None
        if self.state == 'gil':
            code.put_release_ensured_gil(variable)
        else:
            code.put_acquire_gil(variable, unknown_gil_state=not self.scope_gil_state_known)