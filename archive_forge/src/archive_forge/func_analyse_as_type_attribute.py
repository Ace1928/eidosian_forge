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
def analyse_as_type_attribute(self, env):
    if self.obj.is_string_literal:
        return
    type = self.obj.analyse_as_type(env)
    if type:
        if type.is_extension_type or type.is_builtin_type or type.is_cpp_class:
            entry = type.scope.lookup_here(self.attribute)
            if entry and (entry.is_cmethod or (type.is_cpp_class and entry.type.is_cfunction)):
                if type.is_builtin_type:
                    if not self.is_called:
                        return None
                    ubcm_entry = entry
                else:
                    ubcm_entry = self._create_unbound_cmethod_entry(type, entry, env)
                    ubcm_entry.overloaded_alternatives = [self._create_unbound_cmethod_entry(type, overloaded_alternative, env) for overloaded_alternative in entry.overloaded_alternatives]
                return self.as_name_node(env, ubcm_entry, target=False)
        elif type.is_enum or type.is_cpp_enum:
            if self.attribute in type.values:
                for entry in type.entry.enum_values:
                    if entry.name == self.attribute:
                        return self.as_name_node(env, entry, target=False)
                else:
                    error(self.pos, '%s not a known value of %s' % (self.attribute, type))
            else:
                error(self.pos, '%s not a known value of %s' % (self.attribute, type))
    return None