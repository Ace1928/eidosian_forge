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
def analyse_attribute(self, env, obj_type=None):
    immutable_obj = obj_type is not None
    self.is_py_attr = 0
    self.member = self.attribute
    if obj_type is None:
        if self.obj.type.is_string or self.obj.type.is_pyunicode_ptr:
            self.obj = self.obj.coerce_to_pyobject(env)
        obj_type = self.obj.type
    elif obj_type.is_string or obj_type.is_pyunicode_ptr:
        obj_type = py_object_type
    if obj_type.is_ptr or obj_type.is_array:
        obj_type = obj_type.base_type
        self.op = '->'
    elif obj_type.is_extension_type or obj_type.is_builtin_type:
        self.op = '->'
    elif obj_type.is_reference and obj_type.is_fake_reference:
        self.op = '->'
    else:
        self.op = '.'
    if obj_type.has_attributes:
        if obj_type.attributes_known():
            entry = obj_type.scope.lookup_here(self.attribute)
            if obj_type.is_memoryviewslice and (not entry):
                if self.attribute == 'T':
                    self.is_memslice_transpose = True
                    self.is_temp = True
                    self.use_managed_ref = True
                    self.type = self.obj.type.transpose(self.pos)
                    return
                else:
                    obj_type.declare_attribute(self.attribute, env, self.pos)
                    entry = obj_type.scope.lookup_here(self.attribute)
            if entry and entry.is_member:
                entry = None
        else:
            error(self.pos, "Cannot select attribute of incomplete type '%s'" % obj_type)
            self.type = PyrexTypes.error_type
            return
        self.entry = entry
        if entry:
            if obj_type.is_extension_type and entry.name == '__weakref__':
                error(self.pos, 'Illegal use of special attribute __weakref__')
            if entry.is_cproperty:
                self.type = entry.type
                return
            elif entry.is_variable and (not entry.fused_cfunction) or entry.is_cmethod:
                self.type = entry.type
                self.member = entry.cname
                return
            else:
                pass
    self.analyse_as_python_attribute(env, obj_type, immutable_obj)