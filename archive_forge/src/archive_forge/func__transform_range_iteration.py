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
def _transform_range_iteration(self, node, range_function, reversed=False):
    args = range_function.arg_tuple.args
    if len(args) < 3:
        step_pos = range_function.pos
        step_value = 1
        step = ExprNodes.IntNode(step_pos, value='1', constant_result=1)
    else:
        step = args[2]
        step_pos = step.pos
        if not isinstance(step.constant_result, _py_int_types):
            return node
        step_value = step.constant_result
        if step_value == 0:
            return node
        step = ExprNodes.IntNode(step_pos, value=str(step_value), constant_result=step_value)
    if len(args) == 1:
        bound1 = ExprNodes.IntNode(range_function.pos, value='0', constant_result=0)
        bound2 = args[0].coerce_to_integer(self.current_env())
    else:
        bound1 = args[0].coerce_to_integer(self.current_env())
        bound2 = args[1].coerce_to_integer(self.current_env())
    relation1, relation2 = self._find_for_from_node_relations(step_value < 0, reversed)
    bound2_ref_node = None
    if reversed:
        bound1, bound2 = (bound2, bound1)
        abs_step = abs(step_value)
        if abs_step != 1:
            if isinstance(bound1.constant_result, _py_int_types) and isinstance(bound2.constant_result, _py_int_types):
                if step_value < 0:
                    begin_value = bound2.constant_result
                    end_value = bound1.constant_result
                    bound1_value = begin_value - abs_step * ((begin_value - end_value - 1) // abs_step) - 1
                else:
                    begin_value = bound1.constant_result
                    end_value = bound2.constant_result
                    bound1_value = end_value + abs_step * ((begin_value - end_value - 1) // abs_step) + 1
                bound1 = ExprNodes.IntNode(bound1.pos, value=str(bound1_value), constant_result=bound1_value, type=PyrexTypes.spanning_type(bound1.type, bound2.type))
            else:
                bound2_ref_node = UtilNodes.LetRefNode(bound2)
                bound1 = self._build_range_step_calculation(bound1, bound2_ref_node, step, step_value)
    if step_value < 0:
        step_value = -step_value
    step.value = str(step_value)
    step.constant_result = step_value
    step = step.coerce_to_integer(self.current_env())
    if not bound2.is_literal:
        bound2_is_temp = True
        bound2 = bound2_ref_node or UtilNodes.LetRefNode(bound2)
    else:
        bound2_is_temp = False
    for_node = Nodes.ForFromStatNode(node.pos, target=node.target, bound1=bound1, relation1=relation1, relation2=relation2, bound2=bound2, step=step, body=node.body, else_clause=node.else_clause, from_range=True)
    for_node.set_up_loop(self.current_env())
    if bound2_is_temp:
        for_node = UtilNodes.LetNode(bound2, for_node)
    return for_node