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
def _unpack_string_and_cstring_node(self, node):
    if isinstance(node, ExprNodes.CoerceToPyTypeNode):
        node = node.arg
    if isinstance(node, ExprNodes.UnicodeNode):
        encoding = node.value
        node = ExprNodes.BytesNode(node.pos, value=encoding.as_utf8_string(), type=PyrexTypes.c_const_char_ptr_type)
    elif isinstance(node, (ExprNodes.StringNode, ExprNodes.BytesNode)):
        encoding = node.value.decode('ISO-8859-1')
        node = ExprNodes.BytesNode(node.pos, value=node.value, type=PyrexTypes.c_const_char_ptr_type)
    elif node.type is Builtin.bytes_type:
        encoding = None
        node = node.coerce_to(PyrexTypes.c_const_char_ptr_type, self.current_env())
    elif node.type.is_string:
        encoding = None
    else:
        encoding = node = None
    return (encoding, node)