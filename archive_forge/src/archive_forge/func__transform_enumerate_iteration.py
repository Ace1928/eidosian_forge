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
def _transform_enumerate_iteration(self, node, enumerate_function):
    args = enumerate_function.arg_tuple.args
    if len(args) == 0:
        error(enumerate_function.pos, 'enumerate() requires an iterable argument')
        return node
    elif len(args) > 2:
        error(enumerate_function.pos, 'enumerate() takes at most 2 arguments')
        return node
    if not node.target.is_sequence_constructor:
        return node
    targets = node.target.args
    if len(targets) != 2:
        return node
    enumerate_target, iterable_target = targets
    counter_type = enumerate_target.type
    if not counter_type.is_pyobject and (not counter_type.is_int):
        return node
    if len(args) == 2:
        start = unwrap_coerced_node(args[1]).coerce_to(counter_type, self.current_env())
    else:
        start = ExprNodes.IntNode(enumerate_function.pos, value='0', type=counter_type, constant_result=0)
    temp = UtilNodes.LetRefNode(start)
    inc_expression = ExprNodes.AddNode(enumerate_function.pos, operand1=temp, operand2=ExprNodes.IntNode(node.pos, value='1', type=counter_type, constant_result=1), operator='+', type=counter_type, is_temp=counter_type.is_pyobject)
    loop_body = [Nodes.SingleAssignmentNode(pos=enumerate_target.pos, lhs=enumerate_target, rhs=temp), Nodes.SingleAssignmentNode(pos=enumerate_target.pos, lhs=temp, rhs=inc_expression)]
    if isinstance(node.body, Nodes.StatListNode):
        node.body.stats = loop_body + node.body.stats
    else:
        loop_body.append(node.body)
        node.body = Nodes.StatListNode(node.body.pos, stats=loop_body)
    node.target = iterable_target
    node.item = node.item.coerce_to(iterable_target.type, self.current_env())
    node.iterator.sequence = args[0]
    return UtilNodes.LetNode(temp, self._optimise_for_loop(node, node.iterator.sequence))