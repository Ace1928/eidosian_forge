from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def visit_BoolBinopNode(self, node):
    self._calculate_const(node)
    if not node.operand1.has_constant_result():
        return node
    if node.operand1.constant_result:
        if node.operator == 'and':
            return node.operand2
        else:
            return node.operand1
    elif node.operator == 'and':
        return node.operand1
    else:
        return node.operand2