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
class CascadedAssignmentNode(AssignmentNode):
    child_attrs = ['lhs_list', 'rhs', 'coerced_values', 'cloned_values']
    cloned_values = None
    coerced_values = None
    assignment_overloads = None

    def _check_const_assignment(self, node):
        if isinstance(node, CascadedAssignmentNode):
            for lhs in node.lhs_list:
                self._warn_on_const_assignment(lhs, node.rhs)

    def analyse_declarations(self, env):
        for lhs in self.lhs_list:
            lhs.analyse_target_declaration(env)

    def analyse_types(self, env, use_temp=0):
        from .ExprNodes import CloneNode, ProxyNode
        lhs_types = set()
        for i, lhs in enumerate(self.lhs_list):
            lhs = self.lhs_list[i] = lhs.analyse_target_types(env)
            lhs.gil_assignment_check(env)
            lhs_types.add(lhs.type)
        rhs = self.rhs.analyse_types(env)
        if len(lhs_types) == 1:
            if next(iter(lhs_types)).is_cpp_class:
                op = env.lookup_operator('=', [lhs, self.rhs])
                if not op:
                    rhs = rhs.coerce_to(lhs_types.pop(), env)
            else:
                rhs = rhs.coerce_to(lhs_types.pop(), env)
        if not rhs.is_name and (not rhs.is_literal) and (use_temp or rhs.is_attribute or rhs.type.is_pyobject):
            rhs = rhs.coerce_to_temp(env)
        else:
            rhs = rhs.coerce_to_simple(env)
        self.rhs = ProxyNode(rhs) if rhs.is_temp else rhs
        self.coerced_values = []
        coerced_values = {}
        self.assignment_overloads = []
        for lhs in self.lhs_list:
            overloaded = lhs.type.is_cpp_class and env.lookup_operator('=', [lhs, self.rhs])
            self.assignment_overloads.append(overloaded)
            if lhs.type not in coerced_values and lhs.type != rhs.type:
                rhs = CloneNode(self.rhs)
                if not overloaded:
                    rhs = rhs.coerce_to(lhs.type, env)
                self.coerced_values.append(rhs)
                coerced_values[lhs.type] = rhs
        self.cloned_values = []
        for lhs in self.lhs_list:
            rhs = coerced_values.get(lhs.type, self.rhs)
            self.cloned_values.append(CloneNode(rhs))
        return self

    def generate_rhs_evaluation_code(self, code):
        self.rhs.generate_evaluation_code(code)

    def generate_assignment_code(self, code, overloaded_assignment=False):
        for rhs in self.coerced_values:
            rhs.generate_evaluation_code(code)
        for lhs, rhs, overload in zip(self.lhs_list, self.cloned_values, self.assignment_overloads):
            rhs.generate_evaluation_code(code)
            lhs.generate_assignment_code(rhs, code, overloaded_assignment=overload)
        for rhs_value in self.coerced_values:
            rhs_value.generate_disposal_code(code)
            rhs_value.free_temps(code)
        self.rhs.generate_disposal_code(code)
        self.rhs.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.rhs.generate_function_definitions(env, code)

    def annotate(self, code):
        for rhs in self.coerced_values:
            rhs.annotate(code)
        for lhs, rhs in zip(self.lhs_list, self.cloned_values):
            lhs.annotate(code)
            rhs.annotate(code)
        self.rhs.annotate(code)