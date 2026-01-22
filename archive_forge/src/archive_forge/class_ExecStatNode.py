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
class ExecStatNode(StatNode):
    child_attrs = ['args']

    def analyse_expressions(self, env):
        for i, arg in enumerate(self.args):
            arg = arg.analyse_expressions(env)
            arg = arg.coerce_to_pyobject(env)
            self.args[i] = arg
        env.use_utility_code(Builtin.pyexec_utility_code)
        return self
    nogil_check = Node.gil_error
    gil_message = 'Python exec statement'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        args = []
        for arg in self.args:
            arg.generate_evaluation_code(code)
            args.append(arg.py_result())
        args = tuple(args + ['0', '0'][:3 - len(args)])
        temp_result = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
        code.putln('%s = __Pyx_PyExec3(%s, %s, %s);' % ((temp_result,) + args))
        for arg in self.args:
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        code.putln(code.error_goto_if_null(temp_result, self.pos))
        code.put_gotref(temp_result, py_object_type)
        code.put_decref_clear(temp_result, py_object_type)
        code.funcstate.release_temp(temp_result)

    def annotate(self, code):
        for arg in self.args:
            arg.annotate(code)