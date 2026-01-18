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
def visit_AddNode(self, node):
    self._calculate_const(node)
    if node.constant_result is ExprNodes.not_a_constant:
        return node
    if node.operand1.is_string_literal and node.operand2.is_string_literal:
        str1, str2 = (node.operand1, node.operand2)
        if isinstance(str1, ExprNodes.UnicodeNode) and isinstance(str2, ExprNodes.UnicodeNode):
            bytes_value = None
            if str1.bytes_value is not None and str2.bytes_value is not None:
                if str1.bytes_value.encoding == str2.bytes_value.encoding:
                    bytes_value = bytes_literal(str1.bytes_value + str2.bytes_value, str1.bytes_value.encoding)
            string_value = EncodedString(node.constant_result)
            return ExprNodes.UnicodeNode(str1.pos, value=string_value, constant_result=node.constant_result, bytes_value=bytes_value)
        elif isinstance(str1, ExprNodes.BytesNode) and isinstance(str2, ExprNodes.BytesNode):
            if str1.value.encoding == str2.value.encoding:
                bytes_value = bytes_literal(node.constant_result, str1.value.encoding)
                return ExprNodes.BytesNode(str1.pos, value=bytes_value, constant_result=node.constant_result)
    return self.visit_BinopNode(node)