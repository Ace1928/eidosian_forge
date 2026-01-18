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
def unop_node(pos, operator, operand):
    if isinstance(operand, IntNode) and operator == '-':
        return IntNode(pos=operand.pos, value=str(-Utils.str_to_number(operand.value)), longness=operand.longness, unsigned=operand.unsigned)
    elif isinstance(operand, UnopNode) and operand.operator == operator in '+-':
        warning(pos, 'Python has no increment/decrement operator: %s%sx == %s(%sx) == x' % ((operator,) * 4), 5)
    return unop_node_classes[operator](pos, operator=operator, operand=operand)