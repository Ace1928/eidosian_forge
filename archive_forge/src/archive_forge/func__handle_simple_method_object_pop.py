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
def _handle_simple_method_object_pop(self, node, function, args, is_unbound_method, is_list=False):
    """Optimistic optimisation as X.pop([n]) is almost always
        referring to a list.
        """
    if not args:
        return node
    obj = args[0]
    if is_list:
        type_name = 'List'
        obj = obj.as_none_safe_node("'NoneType' object has no attribute '%.30s'", error='PyExc_AttributeError', format_args=['pop'])
    else:
        type_name = 'Object'
    if len(args) == 1:
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py%s_Pop' % type_name, self.PyObject_Pop_func_type, args=[obj], may_return_none=True, is_temp=node.is_temp, utility_code=load_c_utility('pop'))
    elif len(args) == 2:
        index = unwrap_coerced_node(args[1])
        py_index = ExprNodes.NoneNode(index.pos)
        orig_index_type = index.type
        if not index.type.is_int:
            if isinstance(index, ExprNodes.IntNode):
                py_index = index.coerce_to_pyobject(self.current_env())
                index = index.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
            elif is_list:
                if index.type.is_pyobject:
                    py_index = index.coerce_to_simple(self.current_env())
                    index = ExprNodes.CloneNode(py_index)
                index = index.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
            else:
                return node
        elif not PyrexTypes.numeric_type_fits(index.type, PyrexTypes.c_py_ssize_t_type):
            return node
        elif isinstance(index, ExprNodes.IntNode):
            py_index = index.coerce_to_pyobject(self.current_env())
        if not orig_index_type.is_int:
            orig_index_type = index.type
        if not orig_index_type.create_to_py_utility_code(self.current_env()):
            return node
        convert_func = orig_index_type.to_py_function
        conversion_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('intval', orig_index_type, None)])
        return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_Py%s_PopIndex' % type_name, self.PyObject_PopIndex_func_type, args=[obj, py_index, index, ExprNodes.IntNode(index.pos, value=str(orig_index_type.signed and 1 or 0), constant_result=orig_index_type.signed and 1 or 0, type=PyrexTypes.c_int_type), ExprNodes.RawCNameExprNode(index.pos, PyrexTypes.c_void_type, orig_index_type.empty_declaration_code()), ExprNodes.RawCNameExprNode(index.pos, conversion_type, convert_func)], may_return_none=True, is_temp=node.is_temp, utility_code=load_c_utility('pop_index'))
    return node