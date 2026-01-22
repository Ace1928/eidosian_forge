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
def _calculate_constant_seq(self, node, sequence_node, factor):
    if factor.constant_result != 1 and sequence_node.args:
        if isinstance(factor.constant_result, _py_int_types) and factor.constant_result <= 0:
            del sequence_node.args[:]
            sequence_node.mult_factor = None
        elif sequence_node.mult_factor is not None:
            if isinstance(factor.constant_result, _py_int_types) and isinstance(sequence_node.mult_factor.constant_result, _py_int_types):
                value = sequence_node.mult_factor.constant_result * factor.constant_result
                sequence_node.mult_factor = ExprNodes.IntNode(sequence_node.mult_factor.pos, value=str(value), constant_result=value)
            else:
                return self.visit_BinopNode(node)
        else:
            sequence_node.mult_factor = factor
    return sequence_node