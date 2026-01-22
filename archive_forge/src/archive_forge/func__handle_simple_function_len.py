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
def _handle_simple_function_len(self, node, function, pos_args):
    """Replace len(char*) by the equivalent call to strlen(),
        len(Py_UNICODE) by the equivalent Py_UNICODE_strlen() and
        len(known_builtin_type) by an equivalent C-API call.
        """
    if len(pos_args) != 1:
        self._error_wrong_arg_count('len', node, pos_args, 1)
        return node
    arg = pos_args[0]
    if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
        arg = arg.arg
    if arg.type.is_string:
        new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_ssize_strlen', self.Pyx_ssize_strlen_func_type, args=[arg], is_temp=node.is_temp)
    elif arg.type.is_pyunicode_ptr:
        new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py_UNICODE_ssize_strlen', self.Pyx_Py_UNICODE_strlen_func_type, args=[arg], is_temp=node.is_temp, utility_code=UtilityCode.load_cached('ssize_pyunicode_strlen', 'StringTools.c'))
    elif arg.type.is_memoryviewslice:
        func_type = PyrexTypes.CFuncType(PyrexTypes.c_py_ssize_t_type, [PyrexTypes.CFuncTypeArg('memoryviewslice', arg.type, None)], nogil=True)
        new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_MemoryView_Len', func_type, args=[arg], is_temp=node.is_temp)
    elif arg.type.is_pyobject:
        cfunc_name = self._map_to_capi_len_function(arg.type)
        if cfunc_name is None:
            arg_type = arg.type
            if (arg_type.is_extension_type or arg_type.is_builtin_type) and arg_type.entry.qualified_name in self._ext_types_with_pysize:
                cfunc_name = 'Py_SIZE'
            else:
                return node
        arg = arg.as_none_safe_node("object of type 'NoneType' has no len()")
        new_node = ExprNodes.PythonCapiCallNode(node.pos, cfunc_name, self.PyObject_Size_func_type, args=[arg], is_temp=node.is_temp)
    elif arg.type.is_unicode_char:
        return ExprNodes.IntNode(node.pos, value='1', constant_result=1, type=node.type)
    else:
        return node
    if node.type not in (PyrexTypes.c_size_t_type, PyrexTypes.c_py_ssize_t_type):
        new_node = new_node.coerce_to(node.type, self.current_env())
    return new_node