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
class ModuleScope(Scope):
    is_module_scope = 1
    has_import_star = 0
    is_cython_builtin = 0
    old_style_globals = 0
    scope_predefined_names = ['__builtins__', '__name__', '__file__', '__doc__', '__path__', '__spec__', '__loader__', '__package__', '__cached__']

    def __init__(self, name, parent_module, context, is_package=False):
        from . import Builtin
        self.parent_module = parent_module
        outer_scope = Builtin.builtin_scope
        Scope.__init__(self, name, outer_scope, parent_module)
        self.is_package = is_package
        self.module_name = name
        self.module_name = EncodedString(self.module_name)
        self.context = context
        self.module_cname = Naming.module_cname
        self.module_dict_cname = Naming.moddict_cname
        self.method_table_cname = Naming.methtable_cname
        self.doc = ''
        self.doc_cname = Naming.moddoc_cname
        self.utility_code_list = []
        self.module_entries = {}
        self.c_includes = {}
        self.type_names = dict(outer_scope.type_names)
        self.pxd_file_loaded = 0
        self.cimported_modules = []
        self.types_imported = set()
        self.included_files = []
        self.has_extern_class = 0
        self.cached_builtins = []
        self.undeclared_cached_builtins = []
        self.namespace_cname = self.module_cname
        self._cached_tuple_types = {}
        self.process_include(Code.IncludeCode('Python.h', initial=True))

    def qualifying_scope(self):
        return self.parent_module

    def global_scope(self):
        return self

    def lookup(self, name, language_level=None, str_is_str=None):
        entry = self.lookup_here(name)
        if entry is not None:
            return entry
        if language_level is None:
            language_level = self.context.language_level if self.context is not None else 3
        if str_is_str is None:
            str_is_str = language_level == 2 or (self.context is not None and Future.unicode_literals not in self.context.future_directives)
        return self.outer_scope.lookup(name, language_level=language_level, str_is_str=str_is_str)

    def declare_tuple_type(self, pos, components):
        components = tuple(components)
        try:
            ttype = self._cached_tuple_types[components]
        except KeyError:
            ttype = self._cached_tuple_types[components] = PyrexTypes.c_tuple_type(components)
        cname = ttype.cname
        entry = self.lookup_here(cname)
        if not entry:
            scope = StructOrUnionScope(cname)
            for ix, component in enumerate(components):
                scope.declare_var(name='f%s' % ix, type=component, pos=pos)
            struct_entry = self.declare_struct_or_union(cname + '_struct', 'struct', scope, typedef_flag=True, pos=pos, cname=cname)
            self.type_entries.remove(struct_entry)
            ttype.struct_entry = struct_entry
            entry = self.declare_type(cname, ttype, pos, cname)
        ttype.entry = entry
        return entry

    def declare_builtin(self, name, pos):
        if not hasattr(builtins, name) and name not in Code.non_portable_builtins_map and (name not in Code.uncachable_builtins):
            if self.has_import_star:
                entry = self.declare_var(name, py_object_type, pos)
                return entry
            else:
                if Options.error_on_unknown_names:
                    error(pos, 'undeclared name not builtin: %s' % name)
                else:
                    warning(pos, 'undeclared name not builtin: %s' % name, 2)
                entry = self.declare(name, None, py_object_type, pos, 'private')
                entry.is_builtin = 1
                return entry
        if Options.cache_builtins:
            for entry in self.cached_builtins:
                if entry.name == name:
                    return entry
        if name == 'globals' and (not self.old_style_globals):
            return self.outer_scope.lookup('__Pyx_Globals')
        else:
            entry = self.declare(None, None, py_object_type, pos, 'private')
        if Options.cache_builtins and name not in Code.uncachable_builtins:
            entry.is_builtin = 1
            entry.is_const = 1
            entry.name = name
            entry.cname = Naming.builtin_prefix + name
            self.cached_builtins.append(entry)
            self.undeclared_cached_builtins.append(entry)
        else:
            entry.is_builtin = 1
            entry.name = name
        entry.qualified_name = self.builtin_scope().qualify_name(name)
        return entry

    def find_module(self, module_name, pos, relative_level=-1):
        is_relative_import = relative_level is not None and relative_level > 0
        from_module = None
        absolute_fallback = False
        if relative_level is not None and relative_level > 0:
            from_module = self
            top_level = 1 if self.is_package else 0
            while relative_level > top_level and from_module:
                from_module = from_module.parent_module
                relative_level -= 1
        elif relative_level != 0:
            from_module = self.parent_module
            absolute_fallback = True
        module_scope = self.global_scope()
        return module_scope.context.find_module(module_name, from_module=from_module, pos=pos, absolute_fallback=absolute_fallback, relative_import=is_relative_import)

    def find_submodule(self, name, as_package=False):
        if '.' in name:
            name, submodule = name.split('.', 1)
        else:
            submodule = None
        scope = self.lookup_submodule(name)
        if not scope:
            scope = ModuleScope(name, parent_module=self, context=self.context, is_package=True if submodule else as_package)
            self.module_entries[name] = scope
        if submodule:
            scope = scope.find_submodule(submodule, as_package=as_package)
        return scope

    def lookup_submodule(self, name):
        if '.' in name:
            name, submodule = name.split('.', 1)
        else:
            submodule = None
        module = self.module_entries.get(name, None)
        if submodule and module is not None:
            module = module.lookup_submodule(submodule)
        return module

    def add_include_file(self, filename, verbatim_include=None, late=False):
        """
        Add `filename` as include file. Add `verbatim_include` as
        verbatim text in the C file.
        Both `filename` and `verbatim_include` can be `None` or empty.
        """
        inc = Code.IncludeCode(filename, verbatim_include, late=late)
        self.process_include(inc)

    def process_include(self, inc):
        """
        Add `inc`, which is an instance of `IncludeCode`, to this
        `ModuleScope`. This either adds a new element to the
        `c_includes` dict or it updates an existing entry.

        In detail: the values of the dict `self.c_includes` are
        instances of `IncludeCode` containing the code to be put in the
        generated C file. The keys of the dict are needed to ensure
        uniqueness in two ways: if an include file is specified in
        multiple "cdef extern" blocks, only one `#include` statement is
        generated. Second, the same include might occur multiple times
        if we find it through multiple "cimport" paths. So we use the
        generated code (of the form `#include "header.h"`) as dict key.

        If verbatim code does not belong to any include file (i.e. it
        was put in a `cdef extern from *` block), then we use a unique
        dict key: namely, the `sortkey()`.

        One `IncludeCode` object can contain multiple pieces of C code:
        one optional "main piece" for the include file and several other
        pieces for the verbatim code. The `IncludeCode.dict_update`
        method merges the pieces of two different `IncludeCode` objects
        if needed.
        """
        key = inc.mainpiece()
        if key is None:
            key = inc.sortkey()
        inc.dict_update(self.c_includes, key)
        inc = self.c_includes[key]

    def add_imported_module(self, scope):
        if scope not in self.cimported_modules:
            for inc in scope.c_includes.values():
                self.process_include(inc)
            self.cimported_modules.append(scope)
            for m in scope.cimported_modules:
                self.add_imported_module(m)

    def add_imported_entry(self, name, entry, pos):
        if entry.is_pyglobal:
            entry.is_variable = True
        if entry not in self.entries:
            self.entries[name] = entry
        else:
            warning(pos, "'%s' redeclared  " % name, 0)

    def declare_module(self, name, scope, pos):
        entry = self.lookup_here(name)
        if entry:
            if entry.is_pyglobal and entry.as_module is scope:
                return entry
            if not (entry.is_pyglobal and (not entry.as_module)):
                return entry
        else:
            entry = self.declare_var(name, py_object_type, pos)
            entry.is_variable = 0
        entry.as_module = scope
        self.add_imported_module(scope)
        return entry

    def declare_var(self, name, type, pos, cname=None, visibility='private', api=False, in_pxd=False, is_cdef=False, pytyping_modifiers=None):
        if visibility not in ('private', 'public', 'extern'):
            error(pos, 'Module-level variable cannot be declared %s' % visibility)
        self._reject_pytyping_modifiers(pos, pytyping_modifiers, ('typing.Optional',))
        if not is_cdef:
            if type is unspecified_type:
                type = py_object_type
            if not (type.is_pyobject and (not type.is_extension_type)):
                raise InternalError('Non-cdef global variable is not a generic Python object')
        if not cname:
            defining = not in_pxd
            if visibility == 'extern' or (visibility == 'public' and defining):
                cname = name
            else:
                cname = self.mangle(Naming.var_prefix, name)
        entry = self.lookup_here(name)
        if entry and entry.defined_in_pxd:
            if not entry.type.same_as(type):
                if visibility == 'extern' and entry.visibility == 'extern':
                    warning(pos, "Variable '%s' type does not match previous declaration" % name, 1)
                    entry.type = type
            if entry.visibility != 'private':
                mangled_cname = self.mangle(Naming.var_prefix, name)
                if entry.cname == mangled_cname:
                    cname = name
                    entry.cname = name
            if not entry.is_implemented:
                entry.is_implemented = True
                return entry
        entry = Scope.declare_var(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, is_cdef=is_cdef, pytyping_modifiers=pytyping_modifiers)
        if is_cdef:
            entry.is_cglobal = 1
            if entry.type.declaration_value:
                entry.init = entry.type.declaration_value
            self.var_entries.append(entry)
        else:
            entry.is_pyglobal = 1
        if Options.cimport_from_pyx:
            entry.used = 1
        return entry

    def declare_cfunction(self, name, type, pos, cname=None, visibility='private', api=0, in_pxd=0, defining=0, modifiers=(), utility_code=None, overridable=False):
        if not defining and 'inline' in modifiers:
            warning(pos, 'Declarations should not be declared inline.', 1)
        if not cname:
            if visibility == 'extern' or (visibility == 'public' and defining):
                cname = name
            else:
                cname = self.mangle(Naming.func_prefix, name)
        if visibility == 'extern' and type.optional_arg_count:
            error(pos, 'Extern functions cannot have default arguments values.')
        entry = self.lookup_here(name)
        if entry and entry.defined_in_pxd:
            if entry.visibility != 'private':
                mangled_cname = self.mangle(Naming.func_prefix, name)
                if entry.cname == mangled_cname:
                    cname = name
                    entry.cname = cname
                    entry.func_cname = cname
        entry = Scope.declare_cfunction(self, name, type, pos, cname=cname, visibility=visibility, api=api, in_pxd=in_pxd, defining=defining, modifiers=modifiers, utility_code=utility_code, overridable=overridable)
        return entry

    def declare_global(self, name, pos):
        entry = self.lookup_here(name)
        if not entry:
            self.declare_var(name, py_object_type, pos)

    def use_utility_code(self, new_code):
        if new_code is not None:
            self.utility_code_list.append(new_code)

    def use_entry_utility_code(self, entry):
        if entry is None:
            return
        if entry.utility_code:
            self.utility_code_list.append(entry.utility_code)
        if entry.utility_code_definition:
            self.utility_code_list.append(entry.utility_code_definition)

    def declare_c_class(self, name, pos, defining=0, implementing=0, module_name=None, base_type=None, objstruct_cname=None, typeobj_cname=None, typeptr_cname=None, visibility='private', typedef_flag=0, api=0, check_size=None, buffer_defaults=None, shadow=0):
        if typedef_flag and visibility != 'extern':
            if not (visibility == 'public' or api):
                warning(pos, "ctypedef only valid for 'extern' , 'public', and 'api'", 2)
            objtypedef_cname = objstruct_cname
            typedef_flag = 0
        else:
            objtypedef_cname = None
        entry = self.lookup_here(name)
        if entry and (not shadow):
            type = entry.type
            if not (entry.is_type and type.is_extension_type):
                entry = None
            else:
                scope = type.scope
                if typedef_flag and (not scope or scope.defined):
                    self.check_previous_typedef_flag(entry, typedef_flag, pos)
                if scope and scope.defined or (base_type and type.base_type):
                    if base_type and base_type is not type.base_type:
                        error(pos, 'Base type does not match previous declaration')
                if base_type and (not type.base_type):
                    type.base_type = base_type
        if not entry or shadow:
            type = PyrexTypes.PyExtensionType(name, typedef_flag, base_type, visibility == 'extern', check_size=check_size)
            type.pos = pos
            type.buffer_defaults = buffer_defaults
            if objtypedef_cname is not None:
                type.objtypedef_cname = objtypedef_cname
            if visibility == 'extern':
                type.module_name = module_name
            else:
                type.module_name = self.qualified_name
            if typeptr_cname:
                type.typeptr_cname = typeptr_cname
            else:
                type.typeptr_cname = self.mangle(Naming.typeptr_prefix, name)
            entry = self.declare_type(name, type, pos, visibility=visibility, defining=0, shadow=shadow)
            entry.is_cclass = True
            if objstruct_cname:
                type.objstruct_cname = objstruct_cname
            elif not entry.in_cinclude:
                type.objstruct_cname = self.mangle(Naming.objstruct_prefix, name)
            else:
                error(entry.pos, "Object name required for 'public' or 'extern' C class")
            self.attach_var_entry_to_c_class(entry)
            self.c_class_entries.append(entry)
        if not type.scope:
            if defining or implementing:
                scope = CClassScope(name=name, outer_scope=self, visibility=visibility, parent_type=type)
                scope.directives = self.directives.copy()
                if base_type and base_type.scope:
                    scope.declare_inherited_c_attributes(base_type.scope)
                type.set_scope(scope)
                self.type_entries.append(entry)
        elif defining and type.scope.defined:
            error(pos, "C class '%s' already defined" % name)
        elif implementing and type.scope.implemented:
            error(pos, "C class '%s' already implemented" % name)
        if defining:
            entry.defined_in_pxd = 1
        if implementing:
            entry.pos = pos
        if visibility != 'private' and entry.visibility != visibility:
            error(pos, "Class '%s' previously declared as '%s'" % (name, entry.visibility))
        if api:
            entry.api = 1
        if objstruct_cname:
            if type.objstruct_cname and type.objstruct_cname != objstruct_cname:
                error(pos, 'Object struct name differs from previous declaration')
            type.objstruct_cname = objstruct_cname
        if typeobj_cname:
            if type.typeobj_cname and type.typeobj_cname != typeobj_cname:
                error(pos, 'Type object name differs from previous declaration')
            type.typeobj_cname = typeobj_cname
        if self.directives.get('final'):
            entry.type.is_final_type = True
        collection_type = self.directives.get('collection_type')
        if collection_type:
            from .UtilityCode import NonManglingModuleScope
            if not isinstance(self, NonManglingModuleScope):
                error(pos, "'collection_type' is not a public cython directive")
        if collection_type == 'sequence':
            entry.type.has_sequence_flag = True
        entry.used = True
        return entry

    def allocate_vtable_names(self, entry):
        type = entry.type
        if type.base_type and type.base_type.vtabslot_cname:
            type.vtabslot_cname = '%s.%s' % (Naming.obj_base_cname, type.base_type.vtabslot_cname)
        elif type.scope and type.scope.cfunc_entries:
            entry_count = len(type.scope.cfunc_entries)
            base_type = type.base_type
            while base_type:
                if not base_type.scope or entry_count > len(base_type.scope.cfunc_entries):
                    break
                if base_type.is_builtin_type:
                    return
                base_type = base_type.base_type
            type.vtabslot_cname = Naming.vtabslot_cname
        if type.vtabslot_cname:
            type.vtabstruct_cname = self.mangle(Naming.vtabstruct_prefix, entry.name)
            type.vtabptr_cname = self.mangle(Naming.vtabptr_prefix, entry.name)

    def check_c_classes_pxd(self):
        for entry in self.c_class_entries:
            if not entry.type.scope:
                error(entry.pos, "C class '%s' is declared but not defined" % entry.name)

    def check_c_class(self, entry):
        type = entry.type
        name = entry.name
        visibility = entry.visibility
        if not type.scope:
            error(entry.pos, "C class '%s' is declared but not defined" % name)
        if visibility != 'extern' and (not type.typeobj_cname):
            type.typeobj_cname = self.mangle(Naming.typeobj_prefix, name)
        if type.scope:
            for method_entry in type.scope.cfunc_entries:
                if not method_entry.is_inherited and (not method_entry.func_cname):
                    error(method_entry.pos, "C method '%s' is declared but not defined" % method_entry.name)
        if type.vtabslot_cname:
            type.vtable_cname = self.mangle(Naming.vtable_prefix, entry.name)

    def check_c_classes(self):
        debug_check_c_classes = 0
        if debug_check_c_classes:
            print('Scope.check_c_classes: checking scope ' + self.qualified_name)
        for entry in self.c_class_entries:
            if debug_check_c_classes:
                print('...entry %s %s' % (entry.name, entry))
                print('......type = ', entry.type)
                print('......visibility = ', entry.visibility)
            self.check_c_class(entry)

    def check_c_functions(self):
        for name, entry in self.entries.items():
            if entry.is_cfunction:
                if entry.defined_in_pxd and entry.scope is self and (entry.visibility != 'extern') and (not entry.in_cinclude) and (not entry.is_implemented):
                    error(entry.pos, "Non-extern C function '%s' declared but not defined" % name)

    def attach_var_entry_to_c_class(self, entry):
        from . import Builtin
        var_entry = Entry(name=entry.name, type=Builtin.type_type, pos=entry.pos, cname=entry.type.typeptr_cname)
        var_entry.qualified_name = entry.qualified_name
        var_entry.is_variable = 1
        var_entry.is_cglobal = 1
        var_entry.is_readonly = 1
        var_entry.scope = entry.scope
        entry.as_variable = var_entry

    def is_cpp(self):
        return self.cpp

    def infer_types(self):
        from .TypeInference import PyObjectTypeInferer
        PyObjectTypeInferer().infer_types(self)