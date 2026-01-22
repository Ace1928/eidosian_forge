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
def analyse_rvalue_entry(self, env):
    self.analyse_entry(env)
    entry = self.entry
    if entry.is_declared_generic:
        self.result_ctype = py_object_type
    if entry.is_pyglobal or entry.is_builtin:
        if entry.is_builtin and entry.is_const:
            self.is_temp = 0
        else:
            self.is_temp = 1
        self.is_used_as_rvalue = 1
    elif entry.type.is_memoryviewslice:
        self.is_temp = False
        self.is_used_as_rvalue = True
        self.use_managed_ref = True
    return self