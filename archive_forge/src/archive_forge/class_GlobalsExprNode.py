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
class GlobalsExprNode(AtomicExprNode):
    type = dict_type
    is_temp = 1

    def analyse_types(self, env):
        env.use_utility_code(Builtin.globals_utility_code)
        return self
    gil_message = 'Constructing globals dict'

    def may_be_none(self):
        return False

    def generate_result_code(self, code):
        code.putln('%s = __Pyx_Globals(); %s' % (self.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)