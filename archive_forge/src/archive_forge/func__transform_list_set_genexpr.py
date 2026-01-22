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
def _transform_list_set_genexpr(self, node, pos_args, target_type):
    """Replace set(genexpr) and list(genexpr) by an inlined comprehension.
        """
    if len(pos_args) > 1:
        return node
    if not isinstance(pos_args[0], ExprNodes.GeneratorExpressionNode):
        return node
    gen_expr_node = pos_args[0]
    loop_node = gen_expr_node.loop
    yield_statements = _find_yield_statements(loop_node)
    if not yield_statements:
        return node
    result_node = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='set' if target_type is Builtin.set_type else 'list', comprehension_type=target_type)
    for yield_expression, yield_stat_node in yield_statements:
        append_node = ExprNodes.ComprehensionAppendNode(yield_expression.pos, expr=yield_expression, target=result_node.target)
        Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
    return result_node