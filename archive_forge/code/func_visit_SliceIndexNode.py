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
def visit_SliceIndexNode(self, node):
    self._calculate_const(node)
    if node.start is None or node.start.constant_result is None:
        start = node.start = None
    else:
        start = node.start.constant_result
    if node.stop is None or node.stop.constant_result is None:
        stop = node.stop = None
    else:
        stop = node.stop.constant_result
    if node.constant_result is not not_a_constant:
        base = node.base
        if base.is_sequence_constructor and base.mult_factor is None:
            base.args = base.args[start:stop]
            return base
        elif base.is_string_literal:
            base = base.as_sliced_node(start, stop)
            if base is not None:
                return base
    return node