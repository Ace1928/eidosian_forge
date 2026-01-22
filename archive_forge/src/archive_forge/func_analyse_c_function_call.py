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
def analyse_c_function_call(self, env):
    func_type = self.function.type
    if func_type is error_type:
        self.type = error_type
        return
    if func_type.is_cfunction and func_type.is_static_method:
        if self.self and self.self.type.is_extension_type:
            error(self.pos, 'Cannot call a static method on an instance variable.')
        args = self.args
    elif self.self:
        args = [self.self] + self.args
    else:
        args = self.args
    if func_type.is_cpp_class:
        overloaded_entry = self.function.type.scope.lookup('operator()')
        if overloaded_entry is None:
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return
    elif hasattr(self.function, 'entry'):
        overloaded_entry = self.function.entry
    elif self.function.is_subscript and self.function.is_fused_index:
        overloaded_entry = self.function.type.entry
    else:
        overloaded_entry = None
    if overloaded_entry:
        if self.function.type.is_fused:
            functypes = self.function.type.get_all_specialized_function_types()
            alternatives = [f.entry for f in functypes]
        else:
            alternatives = overloaded_entry.all_alternatives()
        entry = PyrexTypes.best_match([arg.type for arg in args], alternatives, self.pos, env, args)
        if not entry:
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return
        entry.used = True
        if not func_type.is_cpp_class:
            self.function.entry = entry
        self.function.type = entry.type
        func_type = self.function_type()
    else:
        entry = None
        func_type = self.function_type()
        if not func_type.is_cfunction:
            error(self.pos, "Calling non-function type '%s'" % func_type)
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return
    max_nargs = len(func_type.args)
    expected_nargs = max_nargs - func_type.optional_arg_count
    actual_nargs = len(args)
    if func_type.optional_arg_count and expected_nargs != actual_nargs:
        self.has_optional_args = 1
        self.is_temp = 1
    if entry and entry.is_cmethod and func_type.args and (not func_type.is_static_method):
        formal_arg = func_type.args[0]
        arg = args[0]
        if formal_arg.not_none:
            if self.self:
                self.self = self.self.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(entry.name) <= 30 else ''), error='PyExc_AttributeError', format_args=[entry.name])
            else:
                arg = arg.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=[entry.name, formal_arg.type.name])
        if self.self:
            if formal_arg.accept_builtin_subtypes:
                arg = CMethodSelfCloneNode(self.self)
            else:
                arg = CloneNode(self.self)
            arg = self.coerced_self = arg.coerce_to(formal_arg.type, env)
        elif formal_arg.type.is_builtin_type:
            arg = arg.coerce_to(formal_arg.type, env)
            if arg.type.is_builtin_type and isinstance(arg, PyTypeTestNode):
                arg.exact_builtin_type = False
        args[0] = arg
    some_args_in_temps = False
    for i in range(min(max_nargs, actual_nargs)):
        formal_arg = func_type.args[i]
        formal_type = formal_arg.type
        arg = args[i].coerce_to(formal_type, env)
        if formal_arg.not_none:
            arg = arg.as_none_safe_node("cannot pass None into a C function argument that is declared 'not None'")
        if arg.is_temp:
            if i > 0:
                some_args_in_temps = True
        elif arg.type.is_pyobject and (not env.nogil):
            if i == 0 and self.self is not None:
                pass
            elif arg.nonlocally_immutable():
                pass
            else:
                if i > 0:
                    some_args_in_temps = True
                arg = arg.coerce_to_temp(env)
        args[i] = arg
    for i in range(max_nargs, actual_nargs):
        arg = args[i]
        if arg.type.is_pyobject:
            if arg.type is str_type:
                arg_ctype = PyrexTypes.c_char_ptr_type
            else:
                arg_ctype = arg.type.default_coerced_ctype()
            if arg_ctype is None:
                error(self.args[i - 1].pos, 'Python object cannot be passed as a varargs parameter')
            else:
                args[i] = arg = arg.coerce_to(arg_ctype, env)
        if arg.is_temp and i > 0:
            some_args_in_temps = True
    if some_args_in_temps:
        for i in range(actual_nargs - 1):
            if i == 0 and self.self is not None:
                continue
            arg = args[i]
            if arg.nonlocally_immutable():
                pass
            elif arg.type.is_cpp_class:
                pass
            elif env.nogil and arg.type.is_pyobject:
                pass
            elif i > 0 or (i == 1 and self.self is not None):
                warning(arg.pos, 'Argument evaluation order in C function call is undefined and may not be as expected', 0)
                break
    self.args[:] = args
    if isinstance(self.function, NewExprNode):
        self.type = PyrexTypes.CPtrType(self.function.class_type)
    else:
        self.type = func_type.return_type
    if self.function.is_name or self.function.is_attribute:
        func_entry = self.function.entry
        if func_entry and (func_entry.utility_code or func_entry.utility_code_definition):
            self.is_temp = 1
    if self.type.is_pyobject:
        self.result_ctype = py_object_type
        self.is_temp = 1
    elif func_type.exception_value is not None or func_type.exception_check:
        self.is_temp = 1
    elif self.type.is_memoryviewslice:
        self.is_temp = 1
    if self.is_temp and self.type.is_reference:
        self.type = PyrexTypes.CFakeReferenceType(self.type.ref_base_type)
    if func_type.exception_check == '+':
        if needs_cpp_exception_conversion(func_type):
            env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
    self.overflowcheck = env.directives['overflowcheck']