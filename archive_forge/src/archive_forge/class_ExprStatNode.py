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
class ExprStatNode(StatNode):
    child_attrs = ['expr']

    def analyse_declarations(self, env):
        from . import ExprNodes
        expr = self.expr
        if isinstance(expr, ExprNodes.GeneralCallNode):
            func = expr.function.as_cython_attribute()
            if func == u'declare':
                args, kwds = expr.explicit_args_kwds()
                if len(args):
                    error(expr.pos, 'Variable names must be specified.')
                for var, type_node in kwds.key_value_pairs:
                    type = type_node.analyse_as_type(env)
                    if type is None:
                        error(type_node.pos, 'Unknown type')
                    else:
                        env.declare_var(var.value, type, var.pos, is_cdef=True)
                self.__class__ = PassStatNode
        elif getattr(expr, 'annotation', None) is not None:
            if expr.is_name:
                expr.declare_from_annotation(env)
                self.__class__ = PassStatNode
            elif expr.is_attribute or expr.is_subscript:
                self.__class__ = PassStatNode

    def analyse_expressions(self, env):
        self.expr.result_is_used = False
        self.expr = self.expr.analyse_expressions(env)
        self.expr.result_is_used = False
        return self

    def nogil_check(self, env):
        if self.expr.type.is_pyobject and self.expr.is_temp:
            self.gil_error()
    gil_message = 'Discarding owned Python object'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.expr.result_is_used = False
        self.expr.generate_evaluation_code(code)
        if not self.expr.is_temp and self.expr.result():
            result = self.expr.result()
            if not self.expr.type.is_void:
                result = '(void)(%s)' % result
            code.putln('%s;' % result)
        self.expr.generate_disposal_code(code)
        self.expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.expr.generate_function_definitions(env, code)

    def annotate(self, code):
        self.expr.annotate(code)