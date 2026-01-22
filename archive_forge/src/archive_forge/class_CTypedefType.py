from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CTypedefType(BaseType):
    is_typedef = 1
    typedef_is_external = 0
    to_py_utility_code = None
    from_py_utility_code = None
    subtypes = ['typedef_base_type']

    def __init__(self, name, base_type, cname, is_external=0, namespace=None):
        assert not base_type.is_complex
        self.typedef_name = name
        self.typedef_cname = cname
        self.typedef_base_type = base_type
        self.typedef_is_external = is_external
        self.typedef_namespace = namespace

    def invalid_value(self):
        return self.typedef_base_type.invalid_value()

    def resolve(self):
        return self.typedef_base_type.resolve()

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = self.typedef_name
        else:
            base_code = public_decl(self.typedef_cname, dll_linkage)
        if self.typedef_namespace is not None and (not pyrex):
            base_code = '%s::%s' % (self.typedef_namespace.empty_declaration_code(), base_code)
        return self.base_declaration_code(base_code, entity_code)

    def as_argument_type(self):
        return self

    def cast_code(self, expr_code):
        if self.typedef_base_type.is_array:
            base_type = self.typedef_base_type.base_type
            return CPtrType(base_type).cast_code(expr_code)
        else:
            return BaseType.cast_code(self, expr_code)

    def specialize(self, values):
        base_type = self.typedef_base_type.specialize(values)
        namespace = self.typedef_namespace.specialize(values) if self.typedef_namespace else None
        if base_type is self.typedef_base_type and namespace is self.typedef_namespace:
            return self
        else:
            return create_typedef_type(self.typedef_name, base_type, self.typedef_cname, 0, namespace)

    def __repr__(self):
        return '<CTypedefType %s>' % self.typedef_cname

    def __str__(self):
        return self.typedef_name

    def _create_utility_code(self, template_utility_code, template_function_name):
        type_name = type_identifier(self.typedef_cname)
        utility_code = template_utility_code.specialize(type=self.typedef_cname, TypeName=type_name)
        function_name = template_function_name % type_name
        return (utility_code, function_name)

    def create_to_py_utility_code(self, env):
        if self.typedef_is_external:
            if not self.to_py_utility_code:
                base_type = self.typedef_base_type
                if type(base_type) is CIntType:
                    self.to_py_function = '__Pyx_PyInt_From_' + self.specialization_name()
                    env.use_utility_code(TempitaUtilityCode.load_cached('CIntToPy', 'TypeConversion.c', context={'TYPE': self.empty_declaration_code(), 'TO_PY_FUNCTION': self.to_py_function}))
                    return True
                elif base_type.is_float:
                    pass
                elif base_type.is_complex:
                    pass
                    pass
                elif base_type.is_cpp_string:
                    cname = '__pyx_convert_PyObject_string_to_py_%s' % type_identifier(self)
                    context = {'cname': cname, 'type': self.typedef_cname}
                    from .UtilityCode import CythonUtilityCode
                    env.use_utility_code(CythonUtilityCode.load('string.to_py', 'CppConvert.pyx', context=context))
                    self.to_py_function = cname
                    return True
            if self.to_py_utility_code:
                env.use_utility_code(self.to_py_utility_code)
                return True
        return self.typedef_base_type.create_to_py_utility_code(env)

    def create_from_py_utility_code(self, env):
        if self.typedef_is_external:
            if not self.from_py_utility_code:
                base_type = self.typedef_base_type
                if type(base_type) is CIntType:
                    self.from_py_function = '__Pyx_PyInt_As_' + self.specialization_name()
                    env.use_utility_code(TempitaUtilityCode.load_cached('CIntFromPy', 'TypeConversion.c', context={'TYPE': self.empty_declaration_code(), 'FROM_PY_FUNCTION': self.from_py_function, 'IS_ENUM': base_type.is_enum}))
                    return True
                elif base_type.is_float:
                    pass
                elif base_type.is_complex:
                    pass
                elif base_type.is_cpp_string:
                    cname = '__pyx_convert_string_from_py_%s' % type_identifier(self)
                    context = {'cname': cname, 'type': self.typedef_cname}
                    from .UtilityCode import CythonUtilityCode
                    env.use_utility_code(CythonUtilityCode.load('string.from_py', 'CppConvert.pyx', context=context))
                    self.from_py_function = cname
                    return True
            if self.from_py_utility_code:
                env.use_utility_code(self.from_py_utility_code)
                return True
        return self.typedef_base_type.create_from_py_utility_code(env)

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        if to_py_function is None:
            to_py_function = self.to_py_function
        return self.typedef_base_type.to_py_call_code(source_code, result_code, result_type, to_py_function)

    def from_py_call_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, special_none_cvalue=None):
        return self.typedef_base_type.from_py_call_code(source_code, result_code, error_pos, code, from_py_function or self.from_py_function, error_condition or self.error_condition(result_code), special_none_cvalue=special_none_cvalue)

    def overflow_check_binop(self, binop, env, const_rhs=False):
        env.use_utility_code(UtilityCode.load('Common', 'Overflow.c'))
        type = self.empty_declaration_code()
        name = self.specialization_name()
        if binop == 'lshift':
            env.use_utility_code(TempitaUtilityCode.load_cached('LeftShift', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'SIGNED': self.signed}))
        else:
            if const_rhs:
                binop += '_const'
            _load_overflow_base(env)
            env.use_utility_code(TempitaUtilityCode.load_cached('SizeCheck', 'Overflow.c', context={'TYPE': type, 'NAME': name}))
            env.use_utility_code(TempitaUtilityCode.load_cached('Binop', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'BINOP': binop}))
        return '__Pyx_%s_%s_checking_overflow' % (binop, name)

    def error_condition(self, result_code):
        if self.typedef_is_external:
            if self.exception_value:
                condition = '(%s == %s)' % (result_code, self.cast_code(self.exception_value))
                if self.exception_check:
                    condition += ' && PyErr_Occurred()'
                return condition
        return self.typedef_base_type.error_condition(result_code)

    def __getattr__(self, name):
        return getattr(self.typedef_base_type, name)

    def py_type_name(self):
        return self.typedef_base_type.py_type_name()

    def can_coerce_to_pyobject(self, env):
        return self.typedef_base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.typedef_base_type.can_coerce_from_pyobject(env)