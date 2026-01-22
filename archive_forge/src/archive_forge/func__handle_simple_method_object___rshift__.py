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
def _handle_simple_method_object___rshift__(self, node, function, args, is_unbound_method):
    if len(args) != 2 or not isinstance(args[1], ExprNodes.IntNode):
        return node
    if not args[1].has_constant_result() or not 1 <= args[1].constant_result <= 63:
        return node
    return self._optimise_num_binop('Rshift', node, function, args, is_unbound_method)