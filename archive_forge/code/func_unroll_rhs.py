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
def unroll_rhs(self, env):
    from . import ExprNodes
    if not isinstance(self.lhs, ExprNodes.TupleNode):
        return
    if any((arg.is_starred for arg in self.lhs.args)):
        return
    unrolled = self.unroll(self.rhs, len(self.lhs.args), env)
    if not unrolled:
        return
    check_node, refs, rhs = unrolled
    return self.unroll_assignments(refs, check_node, self.lhs.args, rhs, env)