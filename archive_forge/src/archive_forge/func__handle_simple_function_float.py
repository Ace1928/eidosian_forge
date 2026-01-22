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
def _handle_simple_function_float(self, node, function, pos_args):
    """Transform float() into either a C type cast or a faster C
        function call.
        """
    if len(pos_args) == 0:
        return ExprNodes.FloatNode(node, value='0.0', constant_result=0.0).coerce_to(Builtin.float_type, self.current_env())
    elif len(pos_args) != 1:
        self._error_wrong_arg_count('float', node, pos_args, '0 or 1')
        return node
    func_arg = pos_args[0]
    if isinstance(func_arg, ExprNodes.CoerceToPyTypeNode):
        func_arg = func_arg.arg
    if func_arg.type is PyrexTypes.c_double_type:
        return func_arg
    elif func_arg.type in (PyrexTypes.c_py_ucs4_type, PyrexTypes.c_py_unicode_type):
        return self._pyucs4_to_number(node, function.name, func_arg)
    elif node.type.assignable_from(func_arg.type) or func_arg.type.is_numeric:
        return ExprNodes.TypecastNode(node.pos, operand=func_arg, type=node.type)
    arg = pos_args[0].as_none_safe_node("float() argument must be a string or a number, not 'NoneType'")
    if func_arg.type is Builtin.bytes_type:
        cfunc_name = '__Pyx_PyBytes_AsDouble'
        utility_code_name = 'pybytes_as_double'
    elif func_arg.type is Builtin.bytearray_type:
        cfunc_name = '__Pyx_PyByteArray_AsDouble'
        utility_code_name = 'pybytes_as_double'
    elif func_arg.type is Builtin.unicode_type:
        cfunc_name = '__Pyx_PyUnicode_AsDouble'
        utility_code_name = 'pyunicode_as_double'
    elif func_arg.type is Builtin.str_type:
        cfunc_name = '__Pyx_PyString_AsDouble'
        utility_code_name = 'pystring_as_double'
    elif func_arg.type is Builtin.long_type:
        cfunc_name = 'PyLong_AsDouble'
    else:
        arg = pos_args[0]
        cfunc_name = '__Pyx_PyObject_AsDouble'
        utility_code_name = 'pyobject_as_double'
    return ExprNodes.PythonCapiCallNode(node.pos, cfunc_name, self.PyObject_AsDouble_func_type, args=[arg], is_temp=node.is_temp, utility_code=load_c_utility(utility_code_name) if utility_code_name else None, py_name='float')