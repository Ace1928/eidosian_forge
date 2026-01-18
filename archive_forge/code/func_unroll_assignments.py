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
def unroll_assignments(self, refs, check_node, lhs_list, rhs_list, env):
    from . import UtilNodes
    assignments = []
    for lhs, rhs in zip(lhs_list, rhs_list):
        assignments.append(SingleAssignmentNode(self.pos, lhs=lhs, rhs=rhs, first=self.first))
    node = ParallelAssignmentNode(pos=self.pos, stats=assignments).analyse_expressions(env)
    if check_node:
        node = StatListNode(pos=self.pos, stats=[check_node, node])
    for ref in refs[::-1]:
        node = UtilNodes.LetNode(ref, node)
    return node