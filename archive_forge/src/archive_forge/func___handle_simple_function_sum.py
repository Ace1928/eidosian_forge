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
def __handle_simple_function_sum(self, node, pos_args):
    """Transform sum(genexpr) into an equivalent inlined aggregation loop.
        """
    if len(pos_args) not in (1, 2):
        return node
    if not isinstance(pos_args[0], (ExprNodes.GeneratorExpressionNode, ExprNodes.ComprehensionNode)):
        return node
    gen_expr_node = pos_args[0]
    loop_node = gen_expr_node.loop
    if isinstance(gen_expr_node, ExprNodes.GeneratorExpressionNode):
        yield_expression, yield_stat_node = _find_single_yield_expression(loop_node)
        yield_expression = None
        if yield_expression is None:
            return node
    else:
        yield_stat_node = gen_expr_node.append
        yield_expression = yield_stat_node.expr
        try:
            if not yield_expression.is_literal or not yield_expression.type.is_int:
                return node
        except AttributeError:
            return node
    if len(pos_args) == 1:
        start = ExprNodes.IntNode(node.pos, value='0', constant_result=0)
    else:
        start = pos_args[1]
    result_ref = UtilNodes.ResultRefNode(pos=node.pos, type=PyrexTypes.py_object_type)
    add_node = Nodes.SingleAssignmentNode(yield_expression.pos, lhs=result_ref, rhs=ExprNodes.binop_node(node.pos, '+', result_ref, yield_expression))
    Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, add_node)
    exec_code = Nodes.StatListNode(node.pos, stats=[Nodes.SingleAssignmentNode(start.pos, lhs=UtilNodes.ResultRefNode(pos=node.pos, expression=result_ref), rhs=start, first=True), loop_node])
    return ExprNodes.InlinedGeneratorExpressionNode(gen_expr_node.pos, loop=exec_code, result_node=result_ref, expr_scope=gen_expr_node.expr_scope, orig_func='sum', has_local_scope=gen_expr_node.has_local_scope)