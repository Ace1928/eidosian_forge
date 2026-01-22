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
class ConsolidateOverflowCheck(Visitor.CythonTransform):
    """
    This class facilitates the sharing of overflow checking among all nodes
    of a nested arithmetic expression.  For example, given the expression
    a*b + c, where a, b, and x are all possibly overflowing ints, the entire
    sequence will be evaluated and the overflow bit checked only at the end.
    """
    overflow_bit_node = None

    def visit_Node(self, node):
        if self.overflow_bit_node is not None:
            saved = self.overflow_bit_node
            self.overflow_bit_node = None
            self.visitchildren(node)
            self.overflow_bit_node = saved
        else:
            self.visitchildren(node)
        return node

    def visit_NumBinopNode(self, node):
        if node.overflow_check and node.overflow_fold:
            top_level_overflow = self.overflow_bit_node is None
            if top_level_overflow:
                self.overflow_bit_node = node
            else:
                node.overflow_bit_node = self.overflow_bit_node
                node.overflow_check = False
            self.visitchildren(node)
            if top_level_overflow:
                self.overflow_bit_node = None
        else:
            self.visitchildren(node)
        return node