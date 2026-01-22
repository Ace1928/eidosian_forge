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
class CVarDefNode(StatNode):
    child_attrs = ['base_type', 'declarators']
    decorators = None
    directive_locals = None

    def analyse_declarations(self, env, dest_scope=None):
        if self.directive_locals is None:
            self.directive_locals = {}
        if not dest_scope:
            dest_scope = env
        self.dest_scope = dest_scope
        if self.declarators:
            templates = self.declarators[0].analyse_templates()
        else:
            templates = None
        if templates is not None:
            if self.visibility != 'extern':
                error(self.pos, 'Only extern functions allowed')
            if len(self.declarators) > 1:
                error(self.declarators[1].pos, "Can't multiply declare template types")
            env = TemplateScope('func_template', env)
            env.directives = env.outer_scope.directives
            for template_param in templates:
                env.declare_type(template_param.name, template_param, self.pos)
        base_type = self.base_type.analyse(env)
        modifiers = None
        if self.base_type.is_templated_type_node:
            modifiers = self.base_type.analyse_pytyping_modifiers(env)
        if base_type.is_fused and (not self.in_pxd) and (env.is_c_class_scope or env.is_module_scope):
            error(self.pos, 'Fused types not allowed here')
            return error_type
        self.entry = None
        visibility = self.visibility
        for declarator in self.declarators:
            if len(self.declarators) > 1 and (not isinstance(declarator, CNameDeclaratorNode)) and env.directives['warn.multiple_declarators']:
                warning(declarator.pos, 'Non-trivial type declarators in shared declaration (e.g. mix of pointers and values). Each pointer declaration should be on its own line.', 1)
            create_extern_wrapper = self.overridable and self.visibility == 'extern' and env.is_module_scope
            if create_extern_wrapper:
                declarator.overridable = False
            if isinstance(declarator, CFuncDeclaratorNode):
                name_declarator, type = declarator.analyse(base_type, env, directive_locals=self.directive_locals, visibility=visibility, in_pxd=self.in_pxd)
            else:
                name_declarator, type = declarator.analyse(base_type, env, visibility=visibility, in_pxd=self.in_pxd)
            if not type.is_complete():
                if not (self.visibility == 'extern' and type.is_array or type.is_memoryviewslice):
                    error(declarator.pos, "Variable type '%s' is incomplete" % type)
            if self.visibility == 'extern' and type.is_pyobject:
                error(declarator.pos, 'Python object cannot be declared extern')
            name = name_declarator.name
            cname = name_declarator.cname
            if name == '':
                error(declarator.pos, 'Missing name in declaration.')
                return
            if type.is_reference and self.visibility != 'extern':
                error(declarator.pos, 'C++ references cannot be declared; use a pointer instead')
            if type.is_rvalue_reference and self.visibility != 'extern':
                error(declarator.pos, 'C++ rvalue-references cannot be declared')
            if type.is_cfunction:
                if 'staticmethod' in env.directives:
                    type.is_static_method = True
                self.entry = dest_scope.declare_cfunction(name, type, declarator.pos, cname=cname, visibility=self.visibility, in_pxd=self.in_pxd, api=self.api, modifiers=self.modifiers, overridable=self.overridable)
                if self.entry is not None:
                    self.entry.directive_locals = copy.copy(self.directive_locals)
                if create_extern_wrapper:
                    self.entry.type.create_to_py_utility_code(env)
                    self.entry.create_wrapper = True
            else:
                if self.overridable:
                    error(self.pos, "Variables cannot be declared with 'cpdef'. Use 'cdef' instead.")
                if self.directive_locals:
                    error(self.pos, 'Decorators can only be followed by functions')
                self.entry = dest_scope.declare_var(name, type, declarator.pos, cname=cname, visibility=visibility, in_pxd=self.in_pxd, api=self.api, is_cdef=True, pytyping_modifiers=modifiers)
                if Options.docstrings:
                    self.entry.doc = embed_position(self.pos, self.doc)