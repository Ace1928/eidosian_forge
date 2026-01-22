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
class _IndexingBaseNode(ExprNode):

    def is_ephemeral(self):
        return self.base.is_ephemeral() or self.base.type in (basestring_type, str_type, bytes_type, bytearray_type, unicode_type)

    def check_const_addr(self):
        return self.base.check_const_addr() and self.index.check_const()

    def is_lvalue(self):
        if self.type.is_reference:
            if self.type.ref_base_type.is_array:
                return False
        elif self.type.is_ptr:
            return True
        return True