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
class CFuncType(CType):
    is_cfunction = 1
    original_sig = None
    cached_specialized_types = None
    from_fused = False
    is_const_method = False
    op_arg_struct = None
    subtypes = ['return_type', 'args']

    def __init__(self, return_type, args, has_varargs=0, exception_value=None, exception_check=0, calling_convention='', nogil=0, with_gil=0, is_overridable=0, optional_arg_count=0, is_const_method=False, is_static_method=False, templates=None, is_strict_signature=False):
        self.return_type = return_type
        self.args = args
        self.has_varargs = has_varargs
        self.optional_arg_count = optional_arg_count
        self.exception_value = exception_value
        self.exception_check = exception_check
        self.calling_convention = calling_convention
        self.nogil = nogil
        self.with_gil = with_gil
        self.is_overridable = is_overridable
        self.is_const_method = is_const_method
        self.is_static_method = is_static_method
        self.templates = templates
        self.is_strict_signature = is_strict_signature

    def __repr__(self):
        arg_reprs = list(map(repr, self.args))
        if self.has_varargs:
            arg_reprs.append('...')
        if self.exception_value:
            except_clause = ' %r' % self.exception_value
        else:
            except_clause = ''
        if self.exception_check:
            except_clause += '?'
        return '<CFuncType %s %s[%s]%s>' % (repr(self.return_type), self.calling_convention_prefix(), ','.join(arg_reprs), except_clause)

    def with_with_gil(self, with_gil):
        if with_gil == self.with_gil:
            return self
        else:
            return CFuncType(self.return_type, self.args, self.has_varargs, self.exception_value, self.exception_check, self.calling_convention, self.nogil, with_gil, self.is_overridable, self.optional_arg_count, self.is_const_method, self.is_static_method, self.templates, self.is_strict_signature)

    def calling_convention_prefix(self):
        cc = self.calling_convention
        if cc:
            return cc + ' '
        else:
            return ''

    def as_argument_type(self):
        return c_ptr_type(self)

    def same_c_signature_as(self, other_type, as_cmethod=0):
        return self.same_c_signature_as_resolved_type(other_type.resolve(), as_cmethod)

    def same_c_signature_as_resolved_type(self, other_type, as_cmethod=False, as_pxd_definition=False, exact_semantics=True):
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        if self.is_overridable != other_type.is_overridable:
            return 0
        nargs = len(self.args)
        if nargs != len(other_type.args):
            return 0
        for i in range(as_cmethod, nargs):
            if not self.args[i].type.same_as(other_type.args[i].type):
                return 0
        if self.has_varargs != other_type.has_varargs:
            return 0
        if self.optional_arg_count != other_type.optional_arg_count:
            return 0
        if as_pxd_definition:
            if not self.return_type.subtype_of_resolved_type(other_type.return_type):
                return 0
        elif not self.return_type.same_as(other_type.return_type):
            return 0
        if not self.same_calling_convention_as(other_type):
            return 0
        if exact_semantics:
            if self.exception_check != other_type.exception_check:
                return 0
            if not self._same_exception_value(other_type.exception_value):
                return 0
        elif not self._is_exception_compatible_with(other_type):
            return 0
        return 1

    def _same_exception_value(self, other_exc_value):
        if self.exception_value == other_exc_value:
            return 1
        if self.exception_check != '+':
            return 0
        if not self.exception_value or not other_exc_value:
            return 0
        if self.exception_value.type != other_exc_value.type:
            return 0
        if self.exception_value.entry and other_exc_value.entry:
            if self.exception_value.entry.cname != other_exc_value.entry.cname:
                return 0
        if self.exception_value.name != other_exc_value.name:
            return 0
        return 1

    def compatible_signature_with(self, other_type, as_cmethod=0):
        return self.compatible_signature_with_resolved_type(other_type.resolve(), as_cmethod)

    def compatible_signature_with_resolved_type(self, other_type, as_cmethod):
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        if not self.is_overridable and other_type.is_overridable:
            return 0
        nargs = len(self.args)
        if nargs - self.optional_arg_count != len(other_type.args) - other_type.optional_arg_count:
            return 0
        if self.optional_arg_count < other_type.optional_arg_count:
            return 0
        for i in range(as_cmethod, len(other_type.args)):
            if not self.args[i].type.same_as(other_type.args[i].type):
                return 0
        if self.has_varargs != other_type.has_varargs:
            return 0
        if not self.return_type.subtype_of_resolved_type(other_type.return_type):
            return 0
        if not self.same_calling_convention_as(other_type):
            return 0
        if self.nogil != other_type.nogil:
            return 0
        if not self._is_exception_compatible_with(other_type):
            return 0
        self.original_sig = other_type.original_sig or other_type
        return 1

    def _is_exception_compatible_with(self, other_type):
        if self.exception_check == '+' and other_type.exception_check != '+':
            return 0
        if not other_type.exception_check or other_type.exception_value is not None:
            if other_type.exception_check and (not (self.exception_check or self.exception_value)):
                return 1
            if not self._same_exception_value(other_type.exception_value):
                return 0
            if self.exception_check and self.exception_check != other_type.exception_check:
                return 0
        return 1

    def narrower_c_signature_than(self, other_type, as_cmethod=0):
        return self.narrower_c_signature_than_resolved_type(other_type.resolve(), as_cmethod)

    def narrower_c_signature_than_resolved_type(self, other_type, as_cmethod):
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        nargs = len(self.args)
        if nargs != len(other_type.args):
            return 0
        for i in range(as_cmethod, nargs):
            if not self.args[i].type.subtype_of_resolved_type(other_type.args[i].type):
                return 0
            else:
                self.args[i].needs_type_test = other_type.args[i].needs_type_test or not self.args[i].type.same_as(other_type.args[i].type)
        if self.has_varargs != other_type.has_varargs:
            return 0
        if self.optional_arg_count != other_type.optional_arg_count:
            return 0
        if not self.return_type.subtype_of_resolved_type(other_type.return_type):
            return 0
        if not self.exception_check and other_type.exception_check:
            return 0
        if not self._same_exception_value(other_type.exception_value):
            return 0
        return 1

    def same_calling_convention_as(self, other):
        sc1 = self.calling_convention == '__stdcall'
        sc2 = other.calling_convention == '__stdcall'
        return sc1 == sc2

    def same_as_resolved_type(self, other_type, as_cmethod=False):
        return self.same_c_signature_as_resolved_type(other_type, as_cmethod=as_cmethod) and self.nogil == other_type.nogil

    def pointer_assignable_from_resolved_type(self, rhs_type):
        if rhs_type is error_type:
            return 1
        if not rhs_type.is_cfunction:
            return 0
        return rhs_type.same_c_signature_as_resolved_type(self, exact_semantics=False) and (not (self.nogil and (not rhs_type.nogil)))

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0, with_calling_convention=1):
        arg_decl_list = []
        for arg in self.args[:len(self.args) - self.optional_arg_count]:
            arg_decl_list.append(arg.type.declaration_code('', for_display, pyrex=pyrex))
        if self.is_overridable:
            arg_decl_list.append('int %s' % Naming.skip_dispatch_cname)
        if self.optional_arg_count:
            if self.op_arg_struct:
                arg_decl_list.append(self.op_arg_struct.declaration_code(Naming.optional_args_cname))
            else:
                assert for_display
        if self.has_varargs:
            arg_decl_list.append('...')
        arg_decl_code = ', '.join(arg_decl_list)
        if not arg_decl_code and (not pyrex):
            arg_decl_code = 'void'
        trailer = ''
        if (pyrex or for_display) and (not self.return_type.is_pyobject):
            if self.exception_value and self.exception_check:
                trailer = ' except? %s' % self.exception_value
            elif self.exception_value and (not self.exception_check):
                trailer = ' except %s' % self.exception_value
            elif not self.exception_value and (not self.exception_check):
                trailer = ' noexcept'
            elif self.exception_check == '+':
                trailer = ' except +'
            elif self.exception_check and for_display:
                trailer = ' except *'
            if self.nogil:
                trailer += ' nogil'
        if not with_calling_convention:
            cc = ''
        else:
            cc = self.calling_convention_prefix()
            if not entity_code and cc or entity_code.startswith('*'):
                entity_code = '(%s%s)' % (cc, entity_code)
                cc = ''
        if self.is_const_method:
            trailer += ' const'
        return self.return_type.declaration_code('%s%s(%s)%s' % (cc, entity_code, arg_decl_code, trailer), for_display, dll_linkage, pyrex)

    def function_header_code(self, func_name, arg_code):
        if self.is_const_method:
            trailer = ' const'
        else:
            trailer = ''
        return '%s%s(%s)%s' % (self.calling_convention_prefix(), func_name, arg_code, trailer)

    def signature_string(self):
        s = self.empty_declaration_code()
        return s

    def signature_cast_string(self):
        s = self.declaration_code('(*)', with_calling_convention=False)
        return '(%s)' % s

    def specialize(self, values):
        result = CFuncType(self.return_type.specialize(values), [arg.specialize(values) for arg in self.args], has_varargs=self.has_varargs, exception_value=self.exception_value, exception_check=self.exception_check, calling_convention=self.calling_convention, nogil=self.nogil, with_gil=self.with_gil, is_overridable=self.is_overridable, optional_arg_count=self.optional_arg_count, is_const_method=self.is_const_method, is_static_method=self.is_static_method, templates=self.templates)
        result.from_fused = self.is_fused
        return result

    def opt_arg_cname(self, arg_name):
        return self.op_arg_struct.base_type.scope.lookup(arg_name).cname

    def get_all_specialized_permutations(self, fused_types=None):
        """
        Permute all the types. For every specific instance of a fused type, we
        want all other specific instances of all other fused types.

        It returns an iterable of two-tuples of the cname that should prefix
        the cname of the function, and a dict mapping any fused types to their
        respective specific types.
        """
        assert self.is_fused
        if fused_types is None:
            fused_types = self.get_fused_types()
        return get_all_specialized_permutations(fused_types)

    def get_all_specialized_function_types(self):
        """
        Get all the specific function types of this one.
        """
        assert self.is_fused
        if self.entry.fused_cfunction:
            return [n.type for n in self.entry.fused_cfunction.nodes]
        elif self.cached_specialized_types is not None:
            return self.cached_specialized_types
        result = []
        permutations = self.get_all_specialized_permutations()
        new_cfunc_entries = []
        for cname, fused_to_specific in permutations:
            new_func_type = self.entry.type.specialize(fused_to_specific)
            if self.optional_arg_count:
                self.declare_opt_arg_struct(new_func_type, cname)
            new_entry = copy.deepcopy(self.entry)
            new_func_type.specialize_entry(new_entry, cname)
            new_entry.type = new_func_type
            new_func_type.entry = new_entry
            result.append(new_func_type)
            new_cfunc_entries.append(new_entry)
        cfunc_entries = self.entry.scope.cfunc_entries
        try:
            cindex = cfunc_entries.index(self.entry)
        except ValueError:
            cfunc_entries.extend(new_cfunc_entries)
        else:
            cfunc_entries[cindex:cindex + 1] = new_cfunc_entries
        self.cached_specialized_types = result
        return result

    def get_fused_types(self, result=None, seen=None, subtypes=None, include_function_return_type=False):
        """Return fused types in the order they appear as parameter types"""
        return super(CFuncType, self).get_fused_types(result, seen, subtypes=self.subtypes if include_function_return_type else ['args'])

    def specialize_entry(self, entry, cname):
        assert not self.is_fused
        specialize_entry(entry, cname)

    def can_coerce_to_pyobject(self, env):
        if self.has_varargs or self.optional_arg_count:
            return False
        if self.to_py_function is not None:
            return self.to_py_function
        for arg in self.args:
            if not arg.type.is_pyobject and (not arg.type.can_coerce_to_pyobject(env)):
                return False
        if not self.return_type.is_pyobject and (not self.return_type.can_coerce_to_pyobject(env)):
            return False
        return True

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return self.to_py_function
        if not self.can_coerce_to_pyobject(env):
            return False
        from .UtilityCode import CythonUtilityCode
        from .Symtab import punycodify_name

        def arg_name_part(arg):
            return '%s%s' % (len(arg.name), punycodify_name(arg.name)) if arg.name else '0'
        arg_names = [arg_name_part(arg) for arg in self.args]
        arg_names = cap_length('_'.join(arg_names))
        safe_typename = type_identifier(self, pyrex=True)
        to_py_function = '__Pyx_CFunc_%s_to_py_%s' % (safe_typename, arg_names)
        for arg in self.args:
            if not arg.type.is_pyobject and (not arg.type.create_from_py_utility_code(env)):
                return False
        if not self.return_type.is_pyobject and (not self.return_type.create_to_py_utility_code(env)):
            return False

        def declared_type(ctype):
            type_displayname = str(ctype.declaration_code('', for_display=True))
            if ctype.is_pyobject:
                arg_ctype = type_name = type_displayname
                if ctype.is_builtin_type:
                    arg_ctype = ctype.name
                elif not ctype.is_extension_type:
                    type_name = 'object'
                    type_displayname = None
                else:
                    type_displayname = repr(type_displayname)
            elif ctype is c_bint_type:
                type_name = arg_ctype = 'bint'
            else:
                type_name = arg_ctype = type_displayname
                if ctype is c_double_type:
                    type_displayname = 'float'
                else:
                    type_displayname = repr(type_displayname)
            return (type_name, arg_ctype, type_displayname)

        class Arg(object):

            def __init__(self, arg_name, arg_type):
                self.name = arg_name
                self.type = arg_type
                self.type_cname, self.ctype, self.type_displayname = declared_type(arg_type)
        if self.return_type.is_void:
            except_clause = 'except *'
        elif self.return_type.is_pyobject:
            except_clause = ''
        elif self.exception_value:
            except_clause = ('except? %s' if self.exception_check else 'except %s') % self.exception_value
        else:
            except_clause = 'except *'
        context = {'cname': to_py_function, 'args': [Arg(arg.name or 'arg%s' % ix, arg.type) for ix, arg in enumerate(self.args)], 'return_type': Arg('return', self.return_type), 'except_clause': except_clause}
        env.use_utility_code(CythonUtilityCode.load('cfunc.to_py', 'CConvert.pyx', outer_module_scope=env.global_scope(), context=context, compiler_directives=dict(env.global_scope().directives)))
        self.to_py_function = to_py_function
        return True