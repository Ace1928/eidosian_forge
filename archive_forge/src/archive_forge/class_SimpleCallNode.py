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
class SimpleCallNode(CallNode):
    subexprs = ['self', 'coerced_self', 'function', 'args', 'arg_tuple']
    self = None
    coerced_self = None
    arg_tuple = None
    wrapper_call = False
    has_optional_args = False
    nogil = False
    analysed = False
    overflowcheck = False

    def compile_time_value(self, denv):
        function = self.function.compile_time_value(denv)
        args = [arg.compile_time_value(denv) for arg in self.args]
        try:
            return function(*args)
        except Exception as e:
            self.compile_time_value_error(e)

    @classmethod
    def for_cproperty(cls, pos, obj, entry):
        property_scope = entry.scope
        getter_entry = property_scope.lookup_here(entry.name)
        assert getter_entry, 'Getter not found in scope %s: %s' % (property_scope, property_scope.entries)
        function = NameNode(pos, name=entry.name, entry=getter_entry, type=getter_entry.type)
        node = cls(pos, function=function, args=[obj])
        return node

    def analyse_as_type(self, env):
        attr = self.function.as_cython_attribute()
        if attr == 'pointer':
            if len(self.args) != 1:
                error(self.args.pos, 'only one type allowed.')
            else:
                type = self.args[0].analyse_as_type(env)
                if not type:
                    error(self.args[0].pos, 'Unknown type')
                else:
                    return PyrexTypes.CPtrType(type)
        elif attr == 'typeof':
            if len(self.args) != 1:
                error(self.args.pos, 'only one type allowed.')
            operand = self.args[0].analyse_types(env)
            return operand.type

    def explicit_args_kwds(self):
        return (self.args, None)

    def analyse_types(self, env):
        if self.analysed:
            return self
        self.analysed = True
        if self.analyse_as_type_constructor(env):
            return self
        self.function.is_called = 1
        self.function = self.function.analyse_types(env)
        function = self.function
        if function.is_attribute and function.entry and function.entry.is_cmethod:
            self.self = function.obj
            function.obj = CloneNode(self.self)
        func_type = self.function_type()
        self.is_numpy_call_with_exprs = False
        if has_np_pythran(env) and function.is_numpy_attribute and pythran_is_numpy_func_supported(function):
            has_pythran_args = True
            self.arg_tuple = TupleNode(self.pos, args=self.args)
            self.arg_tuple = self.arg_tuple.analyse_types(env)
            for arg in self.arg_tuple.args:
                has_pythran_args &= is_pythran_supported_node_or_none(arg)
            self.is_numpy_call_with_exprs = bool(has_pythran_args)
        if self.is_numpy_call_with_exprs:
            env.add_include_file(pythran_get_func_include_file(function))
            return NumPyMethodCallNode.from_node(self, function_cname=pythran_functor(function), arg_tuple=self.arg_tuple, type=PythranExpr(pythran_func_type(function, self.arg_tuple.args)))
        elif func_type.is_pyobject:
            self.arg_tuple = TupleNode(self.pos, args=self.args)
            self.arg_tuple = self.arg_tuple.analyse_types(env).coerce_to_pyobject(env)
            self.args = None
            self.set_py_result_type(function, func_type)
            self.is_temp = 1
        else:
            self.args = [arg.analyse_types(env) for arg in self.args]
            self.analyse_c_function_call(env)
            if func_type.exception_check == '+':
                self.is_temp = True
        return self

    def function_type(self):
        func_type = self.function.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        return func_type

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

    def calculate_result_code(self):
        return self.c_call_code()

    def c_call_code(self):
        func_type = self.function_type()
        if self.type is PyrexTypes.error_type or not func_type.is_cfunction:
            return '<error>'
        formal_args = func_type.args
        arg_list_code = []
        args = list(zip(formal_args, self.args))
        max_nargs = len(func_type.args)
        expected_nargs = max_nargs - func_type.optional_arg_count
        actual_nargs = len(self.args)
        for formal_arg, actual_arg in args[:expected_nargs]:
            arg_code = actual_arg.move_result_rhs_as(formal_arg.type)
            arg_list_code.append(arg_code)
        if func_type.is_overridable:
            arg_list_code.append(str(int(self.wrapper_call or self.function.entry.is_unbound_cmethod)))
        if func_type.optional_arg_count:
            if expected_nargs == actual_nargs:
                optional_args = 'NULL'
            else:
                optional_args = '&%s' % self.opt_arg_struct
            arg_list_code.append(optional_args)
        for actual_arg in self.args[len(formal_args):]:
            arg_list_code.append(actual_arg.move_result_rhs())
        result = '%s(%s)' % (self.function.result(), ', '.join(arg_list_code))
        return result

    def is_c_result_required(self):
        func_type = self.function_type()
        if not func_type.exception_value or func_type.exception_check == '+':
            return False
        return True

    def generate_evaluation_code(self, code):
        function = self.function
        if function.is_name or function.is_attribute:
            code.globalstate.use_entry_utility_code(function.entry)
        abs_function_cnames = ('abs', 'labs', '__Pyx_abs_longlong')
        is_signed_int = self.type.is_int and self.type.signed
        if self.overflowcheck and is_signed_int and (function.result() in abs_function_cnames):
            code.globalstate.use_utility_code(UtilityCode.load_cached('Common', 'Overflow.c'))
            code.putln('if (unlikely(%s == __PYX_MIN(%s))) {                PyErr_SetString(PyExc_OverflowError,                                "Trying to take the absolute value of the most negative integer is not defined."); %s; }' % (self.args[0].result(), self.args[0].type.empty_declaration_code(), code.error_goto(self.pos)))
        if not function.type.is_pyobject or len(self.arg_tuple.args) > 1 or (self.arg_tuple.args and self.arg_tuple.is_literal):
            super(SimpleCallNode, self).generate_evaluation_code(code)
            return
        arg = self.arg_tuple.args[0] if self.arg_tuple.args else None
        subexprs = (self.self, self.coerced_self, function, arg)
        for subexpr in subexprs:
            if subexpr is not None:
                subexpr.generate_evaluation_code(code)
        code.mark_pos(self.pos)
        assert self.is_temp
        self.allocate_temp_result(code)
        if arg is None:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallNoArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallNoArg(%s); %s' % (self.result(), function.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallOneArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallOneArg(%s, %s); %s' % (self.result(), function.py_result(), arg.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        for subexpr in subexprs:
            if subexpr is not None:
                subexpr.generate_disposal_code(code)
                subexpr.free_temps(code)

    def generate_result_code(self, code):
        func_type = self.function_type()
        if func_type.is_pyobject:
            arg_code = self.arg_tuple.py_result()
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCall', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_Call(%s, %s, NULL); %s' % (self.result(), self.function.py_result(), arg_code, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif func_type.is_cfunction:
            nogil = not code.funcstate.gil_owned
            if self.has_optional_args:
                actual_nargs = len(self.args)
                expected_nargs = len(func_type.args) - func_type.optional_arg_count
                self.opt_arg_struct = code.funcstate.allocate_temp(func_type.op_arg_struct.base_type, manage_ref=True)
                code.putln('%s.%s = %s;' % (self.opt_arg_struct, Naming.pyrex_prefix + 'n', len(self.args) - expected_nargs))
                args = list(zip(func_type.args, self.args))
                for formal_arg, actual_arg in args[expected_nargs:actual_nargs]:
                    code.putln('%s.%s = %s;' % (self.opt_arg_struct, func_type.opt_arg_cname(formal_arg.name), actual_arg.result_as(formal_arg.type)))
            exc_checks = []
            if self.type.is_pyobject and self.is_temp:
                exc_checks.append('!%s' % self.result())
            elif self.type.is_memoryviewslice:
                assert self.is_temp
                exc_checks.append(self.type.error_condition(self.result()))
            elif func_type.exception_check != '+':
                exc_val = func_type.exception_value
                exc_check = func_type.exception_check
                if exc_val is not None:
                    exc_checks.append('%s == %s' % (self.result(), func_type.return_type.cast_code(exc_val)))
                if exc_check:
                    if nogil:
                        if not exc_checks:
                            perf_hint_entry = getattr(self.function, 'entry', None)
                            PyrexTypes.write_noexcept_performance_hint(self.pos, code.funcstate.scope, function_name=perf_hint_entry.name if perf_hint_entry else None, void_return=self.type.is_void, is_call=True, is_from_pxd=perf_hint_entry and perf_hint_entry.defined_in_pxd)
                        code.globalstate.use_utility_code(UtilityCode.load_cached('ErrOccurredWithGIL', 'Exceptions.c'))
                        exc_checks.append('__Pyx_ErrOccurredWithGIL()')
                    else:
                        exc_checks.append('PyErr_Occurred()')
            if self.is_temp or exc_checks:
                rhs = self.c_call_code()
                if self.result():
                    lhs = '%s = ' % self.result()
                    if self.is_temp and self.type.is_pyobject:
                        rhs = typecast(py_object_type, self.type, rhs)
                else:
                    lhs = ''
                if func_type.exception_check == '+':
                    translate_cpp_exception(code, self.pos, '%s%s;' % (lhs, rhs), self.result() if self.type.is_pyobject else None, func_type.exception_value, nogil)
                else:
                    if exc_checks:
                        goto_error = code.error_goto_if(' && '.join(exc_checks), self.pos)
                    else:
                        goto_error = ''
                    code.putln('%s%s; %s' % (lhs, rhs, goto_error))
                if self.type.is_pyobject and self.result():
                    self.generate_gotref(code)
            if self.has_optional_args:
                code.funcstate.release_temp(self.opt_arg_struct)