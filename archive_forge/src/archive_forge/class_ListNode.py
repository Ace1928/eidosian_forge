from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class ListNode(SequenceNode):
    obj_conversion_errors = []
    type = list_type
    in_module_scope = False
    gil_message = 'Constructing Python list'

    def type_dependencies(self, env):
        return ()

    def infer_type(self, env):
        return list_type

    def analyse_expressions(self, env):
        for arg in self.args:
            if arg.is_starred:
                arg.starred_expr_allowed_here = True
        node = SequenceNode.analyse_expressions(self, env)
        return node.coerce_to_pyobject(env)

    def analyse_types(self, env):
        with local_errors(ignore=True) as errors:
            self.original_args = list(self.args)
            node = SequenceNode.analyse_types(self, env)
        node.obj_conversion_errors = errors
        if env.is_module_scope:
            self.in_module_scope = True
        node = node._create_merge_node_if_necessary(env)
        return node

    def coerce_to(self, dst_type, env):
        if dst_type.is_pyobject:
            for err in self.obj_conversion_errors:
                report_error(err)
            self.obj_conversion_errors = []
            if not self.type.subtype_of(dst_type):
                error(self.pos, "Cannot coerce list to type '%s'" % dst_type)
        elif (dst_type.is_array or dst_type.is_ptr) and dst_type.base_type is not PyrexTypes.c_void_type:
            array_length = len(self.args)
            if self.mult_factor:
                if isinstance(self.mult_factor.constant_result, _py_int_types):
                    if self.mult_factor.constant_result <= 0:
                        error(self.pos, "Cannot coerce non-positively multiplied list to '%s'" % dst_type)
                    else:
                        array_length *= self.mult_factor.constant_result
                else:
                    error(self.pos, "Cannot coerce dynamically multiplied list to '%s'" % dst_type)
            base_type = dst_type.base_type
            self.type = PyrexTypes.CArrayType(base_type, array_length)
            for i in range(len(self.original_args)):
                arg = self.args[i]
                if isinstance(arg, CoerceToPyTypeNode):
                    arg = arg.arg
                self.args[i] = arg.coerce_to(base_type, env)
        elif dst_type.is_cpp_class:
            return TypecastNode(self.pos, operand=self, type=PyrexTypes.py_object_type).coerce_to(dst_type, env)
        elif self.mult_factor:
            error(self.pos, "Cannot coerce multiplied list to '%s'" % dst_type)
        elif dst_type.is_struct:
            if len(self.args) > len(dst_type.scope.var_entries):
                error(self.pos, "Too many members for '%s'" % dst_type)
            else:
                if len(self.args) < len(dst_type.scope.var_entries):
                    warning(self.pos, "Too few members for '%s'" % dst_type, 1)
                for i, (arg, member) in enumerate(zip(self.original_args, dst_type.scope.var_entries)):
                    if isinstance(arg, CoerceToPyTypeNode):
                        arg = arg.arg
                    self.args[i] = arg.coerce_to(member.type, env)
            self.type = dst_type
        elif dst_type.is_ctuple:
            return self.coerce_to_ctuple(dst_type, env)
        else:
            self.type = error_type
            error(self.pos, "Cannot coerce list to type '%s'" % dst_type)
        return self

    def as_list(self):
        return self

    def as_tuple(self):
        t = TupleNode(self.pos, args=self.args, mult_factor=self.mult_factor)
        if isinstance(self.constant_result, list):
            t.constant_result = tuple(self.constant_result)
        return t

    def allocate_temp_result(self, code):
        if self.type.is_array:
            if self.in_module_scope:
                self.temp_code = code.funcstate.allocate_temp(self.type, manage_ref=False, static=True, reusable=False)
            else:
                self.temp_code = code.funcstate.allocate_temp(self.type, manage_ref=False, reusable=False)
        else:
            SequenceNode.allocate_temp_result(self, code)

    def calculate_constant_result(self):
        if self.mult_factor:
            raise ValueError()
        self.constant_result = [arg.constant_result for arg in self.args]

    def compile_time_value(self, denv):
        l = self.compile_time_value_list(denv)
        if self.mult_factor:
            l *= self.mult_factor.compile_time_value(denv)
        return l

    def generate_operation_code(self, code):
        if self.type.is_pyobject:
            for err in self.obj_conversion_errors:
                report_error(err)
            self.generate_sequence_packing_code(code)
        elif self.type.is_array:
            if self.mult_factor:
                code.putln('{')
                code.putln('Py_ssize_t %s;' % Naming.quick_temp_cname)
                code.putln('for ({i} = 0; {i} < {count}; {i}++) {{'.format(i=Naming.quick_temp_cname, count=self.mult_factor.result()))
                offset = '+ (%d * %s)' % (len(self.args), Naming.quick_temp_cname)
            else:
                offset = ''
            for i, arg in enumerate(self.args):
                if arg.type.is_array:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
                    code.putln('memcpy(&(%s[%s%s]), %s, sizeof(%s[0]));' % (self.result(), i, offset, arg.result(), self.result()))
                else:
                    code.putln('%s[%s%s] = %s;' % (self.result(), i, offset, arg.result()))
            if self.mult_factor:
                code.putln('}')
                code.putln('}')
        elif self.type.is_struct:
            for arg, member in zip(self.args, self.type.scope.var_entries):
                code.putln('%s.%s = %s;' % (self.result(), member.cname, arg.result()))
        else:
            raise InternalError('List type never specified')