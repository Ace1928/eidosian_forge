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
def visit_MergedDictNode(self, node):
    """Unpack **args in place if we can."""
    self.visitchildren(node)
    args = []
    items = []

    def add(parent, arg):
        if arg.is_dict_literal:
            if items and items[-1].reject_duplicates == arg.reject_duplicates:
                items[-1].key_value_pairs.extend(arg.key_value_pairs)
            else:
                items.append(arg)
        elif isinstance(arg, ExprNodes.MergedDictNode) and parent.reject_duplicates == arg.reject_duplicates:
            for child_arg in arg.keyword_args:
                add(arg, child_arg)
        else:
            if items:
                args.extend(items)
                del items[:]
            args.append(arg)
    for arg in node.keyword_args:
        add(node, arg)
    if items:
        args.extend(items)
    if len(args) == 1:
        arg = args[0]
        if arg.is_dict_literal or isinstance(arg, ExprNodes.MergedDictNode):
            return arg
    node.keyword_args[:] = args
    self._calculate_const(node)
    return node