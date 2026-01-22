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
class CallNode(ExprNode):
    may_return_none = None

    def infer_type(self, env):
        function = self.function
        func_type = function.infer_type(env)
        if isinstance(function, NewExprNode):
            return PyrexTypes.CPtrType(function.class_type)
        if func_type is py_object_type:
            entry = getattr(function, 'entry', None)
            if entry is not None:
                func_type = entry.type or func_type
        if func_type.is_ptr:
            func_type = func_type.base_type
        if func_type.is_cfunction:
            if getattr(self.function, 'entry', None) and hasattr(self, 'args'):
                alternatives = self.function.entry.all_alternatives()
                arg_types = [arg.infer_type(env) for arg in self.args]
                func_entry = PyrexTypes.best_match(arg_types, alternatives)
                if func_entry:
                    func_type = func_entry.type
                    if func_type.is_ptr:
                        func_type = func_type.base_type
                    return func_type.return_type
            return func_type.return_type
        elif func_type is type_type:
            if function.is_name and function.entry and function.entry.type:
                result_type = function.entry.type
                if result_type.is_extension_type:
                    return result_type
                elif result_type.is_builtin_type:
                    if function.entry.name == 'float':
                        return PyrexTypes.c_double_type
                    elif function.entry.name in Builtin.types_that_construct_their_instance:
                        return result_type
        func_type = self.function.analyse_as_type(env)
        if func_type and (func_type.is_struct_or_union or func_type.is_cpp_class):
            return func_type
        return py_object_type

    def type_dependencies(self, env):
        return self.function.type_dependencies(env)

    def is_simple(self):
        return False

    def may_be_none(self):
        if self.may_return_none is not None:
            return self.may_return_none
        func_type = self.function.type
        if func_type is type_type and self.function.is_name:
            entry = self.function.entry
            if entry.type.is_extension_type:
                return False
            if entry.type.is_builtin_type and entry.name in Builtin.types_that_construct_their_instance:
                return False
        return ExprNode.may_be_none(self)

    def set_py_result_type(self, function, func_type=None):
        if func_type is None:
            func_type = function.type
        if func_type is Builtin.type_type and (function.is_name and function.entry and function.entry.is_builtin and (function.entry.name in Builtin.types_that_construct_their_instance)):
            if function.entry.name == 'float':
                self.type = PyrexTypes.c_double_type
                self.result_ctype = PyrexTypes.c_double_type
            else:
                self.type = Builtin.builtin_types[function.entry.name]
                self.result_ctype = py_object_type
            self.may_return_none = False
        elif function.is_name and function.type_entry:
            self.type = function.type_entry.type
            self.result_ctype = py_object_type
            self.may_return_none = False
        else:
            self.type = py_object_type

    def analyse_as_type_constructor(self, env):
        type = self.function.analyse_as_type(env)
        if type and type.is_struct_or_union:
            args, kwds = self.explicit_args_kwds()
            items = []
            for arg, member in zip(args, type.scope.var_entries):
                items.append(DictItemNode(pos=arg.pos, key=StringNode(pos=arg.pos, value=member.name), value=arg))
            if kwds:
                items += kwds.key_value_pairs
            self.key_value_pairs = items
            self.__class__ = DictNode
            self.analyse_types(env)
            self.coerce_to(type, env)
            return True
        elif type and type.is_cpp_class:
            self.args = [arg.analyse_types(env) for arg in self.args]
            constructor = type.scope.lookup('<init>')
            if not constructor:
                error(self.function.pos, "no constructor found for C++  type '%s'" % self.function.name)
                self.type = error_type
                return self
            self.function = RawCNameExprNode(self.function.pos, constructor.type)
            self.function.entry = constructor
            self.function.set_cname(type.empty_declaration_code())
            self.analyse_c_function_call(env)
            self.type = type
            return True

    def is_lvalue(self):
        return self.type.is_reference

    def nogil_check(self, env):
        func_type = self.function_type()
        if func_type.is_pyobject:
            self.gil_error()
        elif not func_type.is_error and (not getattr(func_type, 'nogil', False)):
            self.gil_error()
    gil_message = 'Calling gil-requiring function'