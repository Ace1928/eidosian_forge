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
def visit_MulNode(self, node):
    self._calculate_const(node)
    if node.operand1.is_sequence_constructor:
        return self._calculate_constant_seq(node, node.operand1, node.operand2)
    if isinstance(node.operand1, ExprNodes.IntNode) and node.operand2.is_sequence_constructor:
        return self._calculate_constant_seq(node, node.operand2, node.operand1)
    if node.operand1.is_string_literal:
        return self._multiply_string(node, node.operand1, node.operand2)
    elif node.operand2.is_string_literal:
        return self._multiply_string(node, node.operand2, node.operand1)
    return self.visit_BinopNode(node)