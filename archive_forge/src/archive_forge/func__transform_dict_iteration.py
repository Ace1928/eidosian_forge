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
def _transform_dict_iteration(self, node, dict_obj, method, keys, values):
    temps = []
    temp = UtilNodes.TempHandle(PyrexTypes.py_object_type)
    temps.append(temp)
    dict_temp = temp.ref(dict_obj.pos)
    temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    temps.append(temp)
    pos_temp = temp.ref(node.pos)
    key_target = value_target = tuple_target = None
    if keys and values:
        if node.target.is_sequence_constructor:
            if len(node.target.args) == 2:
                key_target, value_target = node.target.args
            else:
                return node
        else:
            tuple_target = node.target
    elif keys:
        key_target = node.target
    else:
        value_target = node.target
    if isinstance(node.body, Nodes.StatListNode):
        body = node.body
    else:
        body = Nodes.StatListNode(pos=node.body.pos, stats=[node.body])
    dict_len_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    temps.append(dict_len_temp)
    dict_len_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=dict_len_temp.ref(dict_obj.pos), type=PyrexTypes.c_ptr_type(dict_len_temp.type))
    temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
    temps.append(temp)
    is_dict_temp = temp.ref(node.pos)
    is_dict_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=is_dict_temp, type=PyrexTypes.c_ptr_type(temp.type))
    iter_next_node = Nodes.DictIterationNextNode(dict_temp, dict_len_temp.ref(dict_obj.pos), pos_temp, key_target, value_target, tuple_target, is_dict_temp)
    iter_next_node = iter_next_node.analyse_expressions(self.current_env())
    body.stats[0:0] = [iter_next_node]
    if method:
        method_node = ExprNodes.StringNode(dict_obj.pos, is_identifier=True, value=method)
        dict_obj = dict_obj.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(method) <= 30 else ''), error='PyExc_AttributeError', format_args=[method])
    else:
        method_node = ExprNodes.NullNode(dict_obj.pos)
        dict_obj = dict_obj.as_none_safe_node("'NoneType' object is not iterable")

    def flag_node(value):
        value = value and 1 or 0
        return ExprNodes.IntNode(node.pos, value=str(value), constant_result=value)
    result_code = [Nodes.SingleAssignmentNode(node.pos, lhs=pos_temp, rhs=ExprNodes.IntNode(node.pos, value='0', constant_result=0)), Nodes.SingleAssignmentNode(dict_obj.pos, lhs=dict_temp, rhs=ExprNodes.PythonCapiCallNode(dict_obj.pos, '__Pyx_dict_iterator', self.PyDict_Iterator_func_type, utility_code=UtilityCode.load_cached('dict_iter', 'Optimize.c'), args=[dict_obj, flag_node(dict_obj.type is Builtin.dict_type), method_node, dict_len_temp_addr, is_dict_temp_addr], is_temp=True)), Nodes.WhileStatNode(node.pos, condition=None, body=body, else_clause=node.else_clause)]
    return UtilNodes.TempsBlockNode(node.pos, temps=temps, body=Nodes.StatListNode(node.pos, stats=result_code))