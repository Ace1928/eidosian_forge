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
class CppClassType(CType):
    is_cpp_class = 1
    has_attributes = 1
    needs_cpp_construction = 1
    exception_check = True
    namespace = None
    kind = 'struct'
    packed = False
    typedef_flag = False
    subtypes = ['templates']

    def __init__(self, name, scope, cname, base_classes, templates=None, template_type=None):
        self.name = name
        self.cname = cname
        self.scope = scope
        self.base_classes = base_classes
        self.operators = []
        self.templates = templates
        self.template_type = template_type
        self.num_optional_templates = sum((is_optional_template_param(T) for T in templates or ()))
        if templates:
            self.specializations = {tuple(zip(templates, templates)): self}
        else:
            self.specializations = {}
        self.is_cpp_string = cname in cpp_string_conversions

    def use_conversion_utility(self, from_or_to):
        pass

    def maybe_unordered(self):
        if 'unordered' in self.cname:
            return 'unordered_'
        else:
            return ''

    def can_coerce_from_pyobject(self, env):
        if self.cname in builtin_cpp_conversions:
            template_count = builtin_cpp_conversions[self.cname]
            for ix, T in enumerate(self.templates or []):
                if ix >= template_count:
                    break
                if T.is_pyobject or not T.can_coerce_from_pyobject(env):
                    return False
            return True
        elif self.cname in cpp_string_conversions:
            return True
        return False

    def create_from_py_utility_code(self, env):
        if self.from_py_function is not None:
            return True
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            X = 'XYZABC'
            tags = []
            context = {}
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if T.is_pyobject or not T.create_from_py_utility_code(env):
                    return False
                tags.append(T.specialization_name())
                context[X[ix]] = T
            if self.cname in cpp_string_conversions:
                cls = 'string'
                tags = (type_identifier(self),)
            else:
                cls = self.cname[5:]
            cname = '__pyx_convert_%s_from_py_%s' % (cls, '__and_'.join(tags))
            context.update({'cname': cname, 'maybe_unordered': self.maybe_unordered(), 'type': self.cname})
            from .UtilityCode import CythonUtilityCode
            directives = CythonUtilityCode.filter_inherited_directives(env.directives)
            env.use_utility_code(CythonUtilityCode.load(cls.replace('unordered_', '') + '.from_py', 'CppConvert.pyx', context=context, compiler_directives=directives))
            self.from_py_function = cname
            return True

    def can_coerce_to_pyobject(self, env):
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if T.is_pyobject or not T.can_coerce_to_pyobject(env):
                    return False
            return True

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return True
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            X = 'XYZABC'
            tags = []
            context = {}
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if not T.create_to_py_utility_code(env):
                    return False
                tags.append(T.specialization_name())
                context[X[ix]] = T
            if self.cname in cpp_string_conversions:
                cls = 'string'
                prefix = 'PyObject_'
                tags = (type_identifier(self),)
            else:
                cls = self.cname[5:]
                prefix = ''
            cname = '__pyx_convert_%s%s_to_py_%s' % (prefix, cls, '____'.join(tags))
            context.update({'cname': cname, 'maybe_unordered': self.maybe_unordered(), 'type': self.cname})
            from .UtilityCode import CythonUtilityCode
            directives = CythonUtilityCode.filter_inherited_directives(env.directives)
            env.use_utility_code(CythonUtilityCode.load(cls.replace('unordered_', '') + '.to_py', 'CppConvert.pyx', context=context, compiler_directives=directives))
            self.to_py_function = cname
            return True

    def is_template_type(self):
        return self.templates is not None and self.template_type is None

    def get_fused_types(self, result=None, seen=None, include_function_return_type=False):
        if result is None:
            result = []
            seen = set()
        if self.namespace:
            self.namespace.get_fused_types(result, seen)
        if self.templates:
            for T in self.templates:
                T.get_fused_types(result, seen)
        return result

    def specialize_here(self, pos, env, template_values=None):
        if not self.is_template_type():
            error(pos, "'%s' type is not a template" % self)
            return error_type
        if len(self.templates) - self.num_optional_templates <= len(template_values) < len(self.templates):
            num_defaults = len(self.templates) - len(template_values)
            partial_specialization = self.declaration_code('', template_params=template_values)
            template_values = template_values + [TemplatePlaceholderType('%s::%s' % (partial_specialization, param.name), True) for param in self.templates[-num_defaults:]]
        if len(self.templates) != len(template_values):
            error(pos, '%s templated type receives %d arguments, got %d' % (self.name, len(self.templates), len(template_values)))
            return error_type
        has_object_template_param = False
        for value in template_values:
            if value.is_pyobject or value.needs_refcounting:
                has_object_template_param = True
                type_description = 'Python object' if value.is_pyobject else 'Reference-counted'
                error(pos, "%s type '%s' cannot be used as a template argument" % (type_description, value))
        if has_object_template_param:
            return error_type
        return self.specialize(dict(zip(self.templates, template_values)))

    def specialize(self, values):
        if not self.templates and (not self.namespace):
            return self
        if self.templates is None:
            self.templates = []
        key = tuple(values.items())
        if key in self.specializations:
            return self.specializations[key]
        template_values = [t.specialize(values) for t in self.templates]
        specialized = self.specializations[key] = CppClassType(self.name, None, self.cname, [], template_values, template_type=self)
        specialized.base_classes = [b.specialize(values) for b in self.base_classes]
        if self.namespace is not None:
            specialized.namespace = self.namespace.specialize(values)
        specialized.scope = self.scope.specialize(values, specialized)
        if self.cname == 'std::vector':
            T = values.get(self.templates[0], None)
            if T and (not T.is_fused) and (T.empty_declaration_code() == 'bool'):
                for bit_ref_returner in ('at', 'back', 'front'):
                    if bit_ref_returner in specialized.scope.entries:
                        specialized.scope.entries[bit_ref_returner].type.return_type = T
        return specialized

    def deduce_template_params(self, actual):
        if actual.is_cv_qualified:
            actual = actual.cv_base_type
        if actual.is_reference:
            actual = actual.ref_base_type
        if self == actual:
            return {}
        elif actual.is_cpp_class:
            self_template_type = self
            while getattr(self_template_type, 'template_type', None):
                self_template_type = self_template_type.template_type

            def all_bases(cls):
                yield cls
                for parent in cls.base_classes:
                    for base in all_bases(parent):
                        yield base
            for actual_base in all_bases(actual):
                template_type = actual_base
                while getattr(template_type, 'template_type', None):
                    template_type = template_type.template_type
                    if self_template_type.empty_declaration_code() == template_type.empty_declaration_code():
                        return reduce(merge_template_deductions, [formal_param.deduce_template_params(actual_param) for formal_param, actual_param in zip(self.templates, actual_base.templates)], {})
        else:
            return {}

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0, template_params=None):
        if template_params is None:
            template_params = self.templates
        if self.templates:
            template_strings = [param.declaration_code('', for_display, None, pyrex) for param in template_params if not is_optional_template_param(param) and (not param.is_fused)]
            if for_display:
                brackets = '[%s]'
            else:
                brackets = '<%s> '
            templates = brackets % ','.join(template_strings)
        else:
            templates = ''
        if pyrex or for_display:
            base_code = '%s%s' % (self.name, templates)
        else:
            base_code = '%s%s' % (self.cname, templates)
            if self.namespace is not None:
                base_code = '%s::%s' % (self.namespace.empty_declaration_code(), base_code)
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def cpp_optional_declaration_code(self, entity_code, dll_linkage=None, template_params=None):
        return '__Pyx_Optional_Type<%s> %s' % (self.declaration_code('', False, dll_linkage, False, template_params), entity_code)

    def is_subclass(self, other_type):
        if self.same_as_resolved_type(other_type):
            return 1
        for base_class in self.base_classes:
            if base_class.is_subclass(other_type):
                return 1
        return 0

    def subclass_dist(self, super_type):
        if self.same_as_resolved_type(super_type):
            return 0
        elif not self.base_classes:
            return float('inf')
        else:
            return 1 + min((b.subclass_dist(super_type) for b in self.base_classes))

    def same_as_resolved_type(self, other_type):
        if other_type.is_cpp_class:
            if self == other_type:
                return 1
            elif self.cname == other_type.cname and (self.template_type and other_type.template_type or self.templates or other_type.templates):
                if self.templates == other_type.templates:
                    return 1
                for t1, t2 in zip(self.templates, other_type.templates):
                    if is_optional_template_param(t1) and is_optional_template_param(t2):
                        break
                    if not t1.same_as_resolved_type(t2):
                        return 0
                return 1
        return 0

    def assignable_from_resolved_type(self, other_type):
        if other_type is error_type:
            return True
        elif other_type.is_cpp_class:
            return other_type.is_subclass(self)
        elif other_type.is_string and self.cname in cpp_string_conversions:
            return True

    def attributes_known(self):
        return self.scope is not None

    def find_cpp_operation_type(self, operator, operand_type=None):
        operands = [self]
        if operand_type is not None:
            operands.append(operand_type)
        operator_entry = self.scope.lookup_operator_for_types(None, operator, operands)
        if not operator_entry:
            return None
        func_type = operator_entry.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        return func_type.return_type

    def get_constructor(self, pos):
        constructor = self.scope.lookup('<init>')
        if constructor is not None:
            return constructor
        nogil = True
        for base in self.base_classes:
            base_constructor = base.scope.lookup('<init>')
            if base_constructor and (not base_constructor.type.nogil):
                nogil = False
                break
        func_type = CFuncType(self, [], exception_check='+', nogil=nogil)
        return self.scope.declare_cfunction(u'<init>', func_type, pos)

    def check_nullary_constructor(self, pos, msg='stack allocated'):
        constructor = self.scope.lookup(u'<init>')
        if constructor is not None and best_match([], constructor.all_alternatives()) is None:
            error(pos, 'C++ class must have a nullary constructor to be %s' % msg)

    def cpp_optional_check_for_null_code(self, cname):
        return '(%s.has_value())' % cname