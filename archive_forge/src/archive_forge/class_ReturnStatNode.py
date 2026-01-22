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
class ReturnStatNode(StatNode):
    child_attrs = ['value']
    is_terminator = True
    in_generator = False
    in_async_gen = False
    in_parallel = False

    def analyse_expressions(self, env):
        return_type = env.return_type
        self.return_type = return_type
        if not return_type:
            error(self.pos, 'Return not inside a function body')
            return self
        if self.value:
            if self.in_async_gen:
                error(self.pos, 'Return with value in async generator')
            self.value = self.value.analyse_types(env)
            if return_type.is_void or return_type.is_returncode:
                error(self.value.pos, 'Return with value in void function')
            else:
                self.value = self.value.coerce_to(env.return_type, env)
        elif not return_type.is_void and (not return_type.is_pyobject) and (not return_type.is_returncode):
            error(self.pos, 'Return value required')
        return self

    def nogil_check(self, env):
        if self.return_type.is_pyobject:
            self.gil_error()
    gil_message = 'Returning Python object'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not self.return_type:
            return
        value = self.value
        if self.return_type.is_pyobject:
            code.put_xdecref(Naming.retval_cname, self.return_type)
            if value and value.is_none:
                value = None
        if value:
            value.generate_evaluation_code(code)
            if self.return_type.is_memoryviewslice:
                from . import MemoryView
                MemoryView.put_acquire_memoryviewslice(lhs_cname=Naming.retval_cname, lhs_type=self.return_type, lhs_pos=value.pos, rhs=value, code=code, have_gil=self.in_nogil_context)
                value.generate_post_assignment_code(code)
            elif self.in_generator:
                code.globalstate.use_utility_code(UtilityCode.load_cached('ReturnWithStopIteration', 'Coroutine.c'))
                code.putln('%s = NULL; __Pyx_ReturnWithStopIteration(%s);' % (Naming.retval_cname, value.py_result()))
                value.generate_disposal_code(code)
            else:
                value.make_owned_reference(code)
                code.putln('%s = %s;' % (Naming.retval_cname, value.result_as(self.return_type)))
                value.generate_post_assignment_code(code)
            value.free_temps(code)
        elif self.return_type.is_pyobject:
            if self.in_generator:
                if self.in_async_gen:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('StopAsyncIteration', 'Coroutine.c'))
                    code.put('PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration); ')
                code.putln('%s = NULL;' % Naming.retval_cname)
            else:
                code.put_init_to_py_none(Naming.retval_cname, self.return_type)
        elif self.return_type.is_returncode:
            self.put_return(code, self.return_type.default_value)
        for cname, type in code.funcstate.temps_holding_reference():
            code.put_decref_clear(cname, type)
        code.put_goto(code.return_label)

    def put_return(self, code, value):
        if self.in_parallel:
            code.putln_openmp('#pragma omp critical(__pyx_returning)')
        code.putln('%s = %s;' % (Naming.retval_cname, value))

    def generate_function_definitions(self, env, code):
        if self.value is not None:
            self.value.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.value:
            self.value.annotate(code)