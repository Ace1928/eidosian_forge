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
def analyse_as_ordinary_attribute_node(self, env, target):
    self.obj = self.obj.analyse_types(env)
    self.analyse_attribute(env)
    if self.entry and self.entry.is_cmethod and (not self.is_called):
        pass
    if self.is_py_attr:
        if not target:
            self.is_temp = 1
            self.result_ctype = py_object_type
    elif target and self.obj.type.is_builtin_type:
        error(self.pos, 'Assignment to an immutable object field')
    elif self.entry and self.entry.is_cproperty:
        if not target:
            return SimpleCallNode.for_cproperty(self.pos, self.obj, self.entry).analyse_types(env)
        error(self.pos, 'Assignment to a read-only property')
    return self