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
class CPtrType(CPointerBaseType):
    is_ptr = 1
    default_value = '0'
    exception_value = 'NULL'

    def __hash__(self):
        return hash(self.base_type) + 27

    def __eq__(self, other):
        if isinstance(other, CType) and other.is_ptr:
            return self.base_type.same_as(other.base_type)
        return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '<CPtrType %s>' % repr(self.base_type)

    def same_as_resolved_type(self, other_type):
        return other_type.is_ptr and self.base_type.same_as(other_type.base_type) or other_type is error_type

    def is_simple_buffer_dtype(self):
        return True

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        return self.base_type.declaration_code('*%s' % entity_code, for_display, dll_linkage, pyrex)

    def assignable_from_resolved_type(self, other_type):
        if other_type is error_type:
            return 1
        if other_type.is_null_ptr:
            return 1
        if self.base_type.is_cv_qualified:
            self = CPtrType(self.base_type.cv_base_type)
        if self.base_type.is_cfunction:
            if other_type.is_ptr:
                other_type = other_type.base_type.resolve()
            if other_type.is_cfunction:
                return self.base_type.pointer_assignable_from_resolved_type(other_type)
            else:
                return 0
        if self.base_type.is_cpp_class and other_type.is_ptr and other_type.base_type.is_cpp_class and other_type.base_type.is_subclass(self.base_type):
            return 1
        if other_type.is_array or other_type.is_ptr:
            return self.base_type.is_void or self.base_type.same_as(other_type.base_type)
        return 0

    def assignment_failure_extra_info(self, src_type, src_name):
        if self.base_type.is_cfunction and src_type.is_ptr:
            src_type = src_type.base_type.resolve()
        if self.base_type.is_cfunction and src_type.is_cfunction:
            copied_src_type = copy.copy(src_type)
            copied_src_type.exception_check = self.base_type.exception_check
            copied_src_type.exception_value = self.base_type.exception_value
            if self.base_type.pointer_assignable_from_resolved_type(copied_src_type):
                msg = 'Exception values are incompatible.'
                if not self.base_type.exception_check and (not self.base_type.exception_value):
                    if src_name is None:
                        src_name = 'the value being assigned'
                    else:
                        src_name = "'{}'".format(src_name)
                    msg += " Suggest adding 'noexcept' to the type of {0}.".format(src_name)
                return msg
        return super(CPtrType, self).assignment_failure_extra_info(src_type, src_name)

    def specialize(self, values):
        base_type = self.base_type.specialize(values)
        if base_type == self.base_type:
            return self
        else:
            return CPtrType(base_type)

    def deduce_template_params(self, actual):
        if isinstance(actual, CPtrType):
            return self.base_type.deduce_template_params(actual.base_type)
        else:
            return {}

    def invalid_value(self):
        return '1'

    def find_cpp_operation_type(self, operator, operand_type=None):
        if self.base_type.is_cpp_class:
            return self.base_type.find_cpp_operation_type(operator, operand_type)
        return None

    def get_fused_types(self, result=None, seen=None, include_function_return_type=False):
        return super(CPointerBaseType, self).get_fused_types(result, seen, include_function_return_type=True)