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
def _transform_reversed_iteration(self, node, reversed_function):
    args = reversed_function.arg_tuple.args
    if len(args) == 0:
        error(reversed_function.pos, 'reversed() requires an iterable argument')
        return node
    elif len(args) > 1:
        error(reversed_function.pos, 'reversed() takes exactly 1 argument')
        return node
    arg = args[0]
    if arg.type in (Builtin.tuple_type, Builtin.list_type):
        node.iterator.sequence = arg.as_none_safe_node("'NoneType' object is not iterable")
        node.iterator.reversed = True
        return node
    return self._optimise_for_loop(node, arg, reversed=True)