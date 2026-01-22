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
def analyse_as_python_attribute(self, env, obj_type=None, immutable_obj=False):
    if obj_type is None:
        obj_type = self.obj.type
    self.attribute = env.mangle_class_private_name(self.attribute)
    self.member = self.attribute
    self.type = py_object_type
    self.is_py_attr = 1
    if not obj_type.is_pyobject and (not obj_type.is_error):
        if obj_type.is_string or obj_type.is_cpp_string or obj_type.is_buffer or obj_type.is_memoryviewslice or obj_type.is_numeric or (obj_type.is_ctuple and obj_type.can_coerce_to_pyobject(env)) or (obj_type.is_struct and obj_type.can_coerce_to_pyobject(env)):
            if not immutable_obj:
                self.obj = self.obj.coerce_to_pyobject(env)
        elif obj_type.is_cfunction and (self.obj.is_name or self.obj.is_attribute) and self.obj.entry.as_variable and self.obj.entry.as_variable.type.is_pyobject:
            if not immutable_obj:
                self.obj = self.obj.coerce_to_pyobject(env)
        else:
            error(self.pos, "Object of type '%s' has no attribute '%s'" % (obj_type, self.attribute))