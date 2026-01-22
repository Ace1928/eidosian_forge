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
def _create_unbound_cmethod_entry(self, type, entry, env):
    if entry.func_cname and entry.type.op_arg_struct is None:
        cname = entry.func_cname
        if entry.type.is_static_method or (env.parent_scope and env.parent_scope.is_cpp_class_scope):
            ctype = entry.type
        elif type.is_cpp_class:
            error(self.pos, '%s not a static member of %s' % (entry.name, type))
            ctype = PyrexTypes.error_type
        else:
            ctype = copy.copy(entry.type)
            ctype.args = ctype.args[:]
            ctype.args[0] = PyrexTypes.CFuncTypeArg('self', type, 'self', None)
    else:
        cname = '%s->%s' % (type.vtabptr_cname, entry.cname)
        ctype = entry.type
    ubcm_entry = Symtab.Entry(entry.name, cname, ctype)
    ubcm_entry.is_cfunction = 1
    ubcm_entry.func_cname = entry.func_cname
    ubcm_entry.is_unbound_cmethod = 1
    ubcm_entry.scope = entry.scope
    return ubcm_entry