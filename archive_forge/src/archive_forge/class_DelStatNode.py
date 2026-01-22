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
class DelStatNode(StatNode):
    child_attrs = ['args']
    ignore_nonexisting = False

    def analyse_declarations(self, env):
        for arg in self.args:
            arg.analyse_target_declaration(env)

    def analyse_expressions(self, env):
        for i, arg in enumerate(self.args):
            arg = self.args[i] = arg.analyse_target_expression(env, None)
            if arg.type.is_pyobject or (arg.is_name and arg.type.is_memoryviewslice):
                if arg.is_name and arg.entry.is_cglobal:
                    error(arg.pos, 'Deletion of global C variable')
            elif arg.type.is_ptr and arg.type.base_type.is_cpp_class:
                self.cpp_check(env)
            elif arg.type.is_cpp_class:
                error(arg.pos, 'Deletion of non-heap C++ object')
            elif arg.is_subscript and arg.base.type is Builtin.bytearray_type:
                pass
            else:
                error(arg.pos, 'Deletion of non-Python, non-C++ object')
        return self

    def nogil_check(self, env):
        for arg in self.args:
            if arg.type.is_pyobject:
                self.gil_error()
    gil_message = 'Deleting Python object'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        for arg in self.args:
            if arg.type.is_pyobject or arg.type.is_memoryviewslice or (arg.is_subscript and arg.base.type is Builtin.bytearray_type):
                arg.generate_deletion_code(code, ignore_nonexisting=self.ignore_nonexisting)
            elif arg.type.is_ptr and arg.type.base_type.is_cpp_class:
                arg.generate_evaluation_code(code)
                code.putln('delete %s;' % arg.result())
                arg.generate_disposal_code(code)
                arg.free_temps(code)

    def annotate(self, code):
        for arg in self.args:
            arg.annotate(code)