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
def wrap_obj_in_nonecheck(self, env):
    if not env.directives['nonecheck']:
        return
    msg = None
    format_args = ()
    if self.obj.type.is_extension_type and self.needs_none_check and (not self.is_py_attr):
        msg = "'NoneType' object has no attribute '%{0}s'".format('.30' if len(self.attribute) <= 30 else '')
        format_args = (self.attribute,)
    elif self.obj.type.is_memoryviewslice:
        if self.is_memslice_transpose:
            msg = 'Cannot transpose None memoryview slice'
        else:
            entry = self.obj.type.scope.lookup_here(self.attribute)
            if entry:
                msg = "Cannot access '%s' attribute of None memoryview slice"
                format_args = (entry.name,)
    if msg:
        self.obj = self.obj.as_none_safe_node(msg, 'PyExc_AttributeError', format_args=format_args)