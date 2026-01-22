from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class CClassScope(ClassScope):
    is_c_class_scope = 1
    is_closure_class_scope = False
    has_pyobject_attrs = False
    has_memoryview_attrs = False
    has_cpp_constructable_attrs = False
    has_cyclic_pyobject_attrs = False
    defined = False
    implemented = False

    def __init__(self, name, outer_scope, visibility, parent_type):
        ClassScope.__init__(self, name, outer_scope)
        if visibility != 'extern':
            self.method_table_cname = outer_scope.mangle(Naming.methtab_prefix, name)
            self.getset_table_cname = outer_scope.mangle(Naming.gstab_prefix, name)
        self.property_entries = []
        self.inherited_var_entries = []
        self.parent_type = parent_type
        if (parent_type.is_builtin_type or parent_type.is_extension_type) and parent_type.typeptr_cname:
            self.namespace_cname = '(PyObject *)%s' % parent_type.typeptr_cname

    def needs_gc(self):
        if self.has_cyclic_pyobject_attrs and (not self.directives.get('no_gc', False)):
            return True
        base_type = self.parent_type.base_type
        if base_type and base_type.scope is not None:
            return base_type.scope.needs_gc()
        elif self.parent_type.is_builtin_type:
            return not self.parent_type.is_gc_simple
        return False

    def needs_trashcan(self):
        directive = self.directives.get('trashcan')
        if directive is False:
            return False
        if directive and self.has_cyclic_pyobject_attrs:
            return True
        base_type = self.parent_type.base_type
        if base_type and base_type.scope is not None:
            return base_type.scope.needs_trashcan()
        return self.parent_type.builtin_trashcan

    def needs_tp_clear(self):
        """
        Do we need to generate an implementation for the tp_clear slot? Can
        be disabled to keep references for the __dealloc__ cleanup function.
        """
        return self.needs_gc() and (not self.directives.get('no_gc_clear', False))

    def may_have_finalize(self):
        """
        This covers cases where we definitely have a __del__ function
        and also cases where one of the base classes could have a __del__
        function but we don't know.
        """
        current_type_scope = self
        while current_type_scope:
            del_entry = current_type_scope.lookup_here('__del__')
            if del_entry and del_entry.is_special:
                return True
            if current_type_scope.parent_type.is_external or not current_type_scope.implemented or current_type_scope.parent_type.multiple_bases:
                return True
            current_base_type = current_type_scope.parent_type.base_type
            current_type_scope = current_base_type.scope if current_base_type else None
        return False

    def get_refcounted_entries(self, include_weakref=False, include_gc_simple=True):
        py_attrs = []
        py_buffers = []
        memoryview_slices = []
        for entry in self.var_entries:
            if entry.type.is_pyobject:
                if include_weakref or (self.is_closure_class_scope or entry.name != '__weakref__'):
                    if include_gc_simple or not entry.type.is_gc_simple:
                        py_attrs.append(entry)
            elif entry.type == PyrexTypes.c_py_buffer_type:
                py_buffers.append(entry)
            elif entry.type.is_memoryviewslice:
                memoryview_slices.append(entry)
        have_entries = py_attrs or py_buffers or memoryview_slices
        return (have_entries, (py_attrs, py_buffers, memoryview_slices))

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        name = self.mangle_class_private_name(name)
        if pytyping_modifiers:
            if 'typing.ClassVar' in pytyping_modifiers:
                is_cdef = 0
                if not type.is_pyobject:
                    if not type.equivalent_type:
                        warning(pos, "ClassVar[] requires the type to be a Python object type. Found '%s', using object instead." % type)
                        type = py_object_type
                    else:
                        type = type.equivalent_type
            if 'dataclasses.InitVar' in pytyping_modifiers and (not self.is_c_dataclass_scope):
                error(pos, 'Use of cython.dataclasses.InitVar does not make sense outside a dataclass')
        if is_cdef:
            if self.defined:
                error(pos, 'C attributes cannot be added in implementation part of extension type defined in a pxd')
            if not self.is_closure_class_scope and get_slot_table(self.directives).get_special_method_signature(name):
                error(pos, "The name '%s' is reserved for a special method." % name)
            if not cname:
                cname = name
                if visibility == 'private':
                    cname = c_safe_identifier(cname)
                cname = punycodify_name(cname, Naming.unicode_structmember_prefix)
            entry = self.declare(name, cname, type, pos, visibility)
            entry.is_variable = 1
            self.var_entries.append(entry)
            entry.pytyping_modifiers = pytyping_modifiers
            if type.is_cpp_class and visibility != 'extern':
                if self.directives['cpp_locals']:
                    entry.make_cpp_optional()
                else:
                    type.check_nullary_constructor(pos)
            if type.is_memoryviewslice:
                self.has_memoryview_attrs = True
            elif type.needs_cpp_construction:
                self.use_utility_code(Code.UtilityCode('#include <new>'))
                self.has_cpp_constructable_attrs = True
            elif type.is_pyobject and (self.is_closure_class_scope or name != '__weakref__'):
                self.has_pyobject_attrs = True
                if not type.is_builtin_type or not type.scope or type.scope.needs_gc():
                    self.has_cyclic_pyobject_attrs = True
            if visibility not in ('private', 'public', 'readonly'):
                error(pos, 'Attribute of extension type cannot be declared %s' % visibility)
            if visibility in ('public', 'readonly'):
                entry.needs_property = True
                if not self.is_closure_class_scope and name == '__weakref__':
                    error(pos, 'Special attribute __weakref__ cannot be exposed to Python')
                if not (type.is_pyobject or type.can_coerce_to_pyobject(self)):
                    error(pos, "C attribute of type '%s' cannot be accessed from Python" % type)
            else:
                entry.needs_property = False
            return entry
        else:
            if type is unspecified_type:
                type = py_object_type
            entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
            entry.is_member = 1
            entry.is_pyglobal = 1
            return entry

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        if name in richcmp_special_methods:
            if self.lookup_here('__richcmp__'):
                error(pos, 'Cannot define both % and __richcmp__' % name)
        elif name == '__richcmp__':
            for n in richcmp_special_methods:
                if self.lookup_here(n):
                    error(pos, 'Cannot define both % and __richcmp__' % n)
        if name == '__new__':
            error(pos, '__new__ method of extension type will change semantics in a future version of Pyrex and Cython. Use __cinit__ instead.')
        entry = self.declare_var(name, py_object_type, pos, visibility='extern')
        special_sig = get_slot_table(self.directives).get_special_method_signature(name)
        if special_sig:
            entry.signature = special_sig
            entry.is_special = 1
        else:
            entry.signature = pymethod_signature
            entry.is_special = 0
        self.pyfunc_entries.append(entry)
        return entry

    def lookup_here(self, name):
        if not self.is_closure_class_scope and name == '__new__':
            name = EncodedString('__cinit__')
        entry = ClassScope.lookup_here(self, name)
        if entry and entry.is_builtin_cmethod:
            if not self.parent_type.is_builtin_type:
                if not self.parent_type.is_final_type:
                    return None
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        name = self.mangle_class_private_name(name)
        if get_slot_table(self.directives).get_special_method_signature(name) and (not self.parent_type.is_builtin_type):
            error(pos, "Special methods must be declared with 'def', not 'cdef'")
        args = type.args
        if not type.is_static_method:
            if not args:
                error(pos, 'C method has no self argument')
            elif not self.parent_type.assignable_from(args[0].type):
                error(pos, "Self argument (%s) of C method '%s' does not match parent type (%s)" % (args[0].type, name, self.parent_type))
        entry = self.lookup_here(name)
        if cname is None:
            cname = punycodify_name(c_safe_identifier(name), Naming.unicode_vtabentry_prefix)
        if entry:
            if not entry.is_cfunction:
                error(pos, "'%s' redeclared " % name)
                entry.already_declared_here()
            else:
                if defining and entry.func_cname:
                    error(pos, "'%s' already defined" % name)
                if entry.is_final_cmethod and entry.is_inherited:
                    error(pos, 'Overriding final methods is not allowed')
                elif type.same_c_signature_as(entry.type, as_cmethod=1) and type.nogil == entry.type.nogil:
                    entry.type = entry.type.with_with_gil(type.with_gil)
                elif type.compatible_signature_with(entry.type, as_cmethod=1) and type.nogil == entry.type.nogil:
                    if self.defined and (not in_pxd) and (not type.same_c_signature_as_resolved_type(entry.type, as_cmethod=1, as_pxd_definition=1)):
                        warning(pos, "Compatible but non-identical C method '%s' not redeclared in definition part of extension type '%s'.  This may cause incorrect vtables to be generated." % (name, self.class_name), 2)
                        warning(entry.pos, 'Previous declaration is here', 2)
                    entry = self.add_cfunction(name, type, pos, cname, visibility='ignore', modifiers=modifiers)
                else:
                    error(pos, 'Signature not compatible with previous declaration')
                    error(entry.pos, 'Previous declaration is here')
        else:
            if self.defined:
                error(pos, "C method '%s' not previously declared in definition part of extension type '%s'" % (name, self.class_name))
            entry = self.add_cfunction(name, type, pos, cname, visibility, modifiers)
        if defining:
            entry.func_cname = self.mangle(Naming.func_prefix, name)
        entry.utility_code = utility_code
        type.entry = entry
        if u'inline' in modifiers:
            entry.is_inline_cmethod = True
        if self.parent_type.is_final_type or entry.is_inline_cmethod or self.directives.get('final'):
            entry.is_final_cmethod = True
            entry.final_func_cname = entry.func_cname
            if not type.is_fused:
                entry.vtable_type = entry.type
                entry.type = type
        return entry

    def add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=False):
        prev_entry = self.lookup_here(name)
        entry = ClassScope.add_cfunction(self, name, type, pos, cname, visibility, modifiers, inherited=inherited)
        entry.is_cmethod = 1
        entry.prev_entry = prev_entry
        return entry

    def declare_builtin_cfunction(self, name, type, cname, utility_code=None):
        name = EncodedString(name)
        entry = self.declare_cfunction(name, type, pos=None, cname=cname, visibility='extern', utility_code=utility_code)
        var_entry = Entry(name, name, py_object_type)
        var_entry.qualified_name = name
        var_entry.is_variable = 1
        var_entry.is_builtin = 1
        var_entry.utility_code = utility_code
        var_entry.scope = entry.scope
        entry.as_variable = var_entry
        return entry

    def declare_property(self, name, doc, pos, ctype=None, property_scope=None):
        entry = self.lookup_here(name)
        if entry is None:
            entry = self.declare(name, name, py_object_type if ctype is None else ctype, pos, 'private')
        entry.is_property = True
        if ctype is not None:
            entry.is_cproperty = True
        entry.doc = doc
        if property_scope is None:
            entry.scope = PropertyScope(name, class_scope=self)
        else:
            entry.scope = property_scope
        self.property_entries.append(entry)
        return entry

    def declare_cproperty(self, name, type, cfunc_name, doc=None, pos=None, visibility='extern', nogil=False, with_gil=False, exception_value=None, exception_check=False, utility_code=None):
        """Internal convenience method to declare a C property function in one go.
        """
        property_entry = self.declare_property(name, doc=doc, ctype=type, pos=pos)
        cfunc_entry = property_entry.scope.declare_cfunction(name=name, type=PyrexTypes.CFuncType(type, [PyrexTypes.CFuncTypeArg('self', self.parent_type, pos=None)], nogil=nogil, with_gil=with_gil, exception_value=exception_value, exception_check=exception_check), cname=cfunc_name, utility_code=utility_code, visibility=visibility, pos=pos)
        return (property_entry, cfunc_entry)

    def declare_inherited_c_attributes(self, base_scope):

        def adapt(cname):
            return '%s.%s' % (Naming.obj_base_cname, base_entry.cname)
        entries = base_scope.inherited_var_entries + base_scope.var_entries
        for base_entry in entries:
            entry = self.declare(base_entry.name, adapt(base_entry.cname), base_entry.type, None, 'private')
            entry.is_variable = 1
            entry.is_inherited = True
            entry.annotation = base_entry.annotation
            self.inherited_var_entries.append(entry)
        for base_entry in base_scope.cfunc_entries[:]:
            if base_entry.type.is_fused:
                base_entry.type.get_all_specialized_function_types()
        for base_entry in base_scope.cfunc_entries:
            cname = base_entry.cname
            var_entry = base_entry.as_variable
            is_builtin = var_entry and var_entry.is_builtin
            if not is_builtin:
                cname = adapt(cname)
            entry = self.add_cfunction(base_entry.name, base_entry.type, base_entry.pos, cname, base_entry.visibility, base_entry.func_modifiers, inherited=True)
            entry.is_inherited = 1
            if base_entry.is_final_cmethod:
                entry.is_final_cmethod = True
                entry.is_inline_cmethod = base_entry.is_inline_cmethod
                if self.parent_scope == base_scope.parent_scope or entry.is_inline_cmethod:
                    entry.final_func_cname = base_entry.final_func_cname
            if is_builtin:
                entry.is_builtin_cmethod = True
                entry.as_variable = var_entry
            if base_entry.utility_code:
                entry.utility_code = base_entry.utility_code