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
def _handle_simple_method_bytearray_append(self, node, function, args, is_unbound_method):
    if len(args) != 2:
        return node
    func_name = '__Pyx_PyByteArray_Append'
    func_type = self.PyByteArray_Append_func_type
    value = unwrap_coerced_node(args[1])
    if value.type.is_int or isinstance(value, ExprNodes.IntNode):
        value = value.coerce_to(PyrexTypes.c_int_type, self.current_env())
        utility_code = UtilityCode.load_cached('ByteArrayAppend', 'StringTools.c')
    elif value.is_string_literal:
        if not value.can_coerce_to_char_literal():
            return node
        value = value.coerce_to(PyrexTypes.c_char_type, self.current_env())
        utility_code = UtilityCode.load_cached('ByteArrayAppend', 'StringTools.c')
    elif value.type.is_pyobject:
        func_name = '__Pyx_PyByteArray_AppendObject'
        func_type = self.PyByteArray_AppendObject_func_type
        utility_code = UtilityCode.load_cached('ByteArrayAppendObject', 'StringTools.c')
    else:
        return node
    new_node = ExprNodes.PythonCapiCallNode(node.pos, func_name, func_type, args=[args[0], value], may_return_none=False, is_temp=node.is_temp, utility_code=utility_code)
    if node.result_is_used:
        new_node = new_node.coerce_to(node.type, self.current_env())
    return new_node