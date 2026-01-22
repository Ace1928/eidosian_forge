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
class ReraiseStatNode(StatNode):
    child_attrs = []
    is_terminator = True

    def analyse_expressions(self, env):
        return self
    nogil_check = Node.gil_error
    gil_message = 'Raising exception'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        vars = code.funcstate.exc_vars
        if vars:
            code.globalstate.use_utility_code(restore_exception_utility_code)
            code.put_giveref(vars[0], py_object_type)
            code.put_giveref(vars[1], py_object_type)
            code.put_xgiveref(vars[2], py_object_type)
            code.putln('__Pyx_ErrRestoreWithState(%s, %s, %s);' % tuple(vars))
            for varname in vars:
                code.put('%s = 0; ' % varname)
            code.putln()
            code.putln(code.error_goto(self.pos))
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ReRaiseException', 'Exceptions.c'))
            code.putln('__Pyx_ReraiseException(); %s' % code.error_goto(self.pos))