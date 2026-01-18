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
def visit_JoinedStrNode(self, node):
    """
        Clean up after the parser by discarding empty Unicode strings and merging
        substring sequences.  Empty or single-value join lists are not uncommon
        because f-string format specs are always parsed into JoinedStrNodes.
        """
    self.visitchildren(node)
    unicode_node = ExprNodes.UnicodeNode
    values = []
    for is_unode_group, substrings in itertools.groupby(node.values, lambda v: isinstance(v, unicode_node)):
        if is_unode_group:
            substrings = list(substrings)
            unode = substrings[0]
            if len(substrings) > 1:
                value = EncodedString(u''.join((value.value for value in substrings)))
                unode = ExprNodes.UnicodeNode(unode.pos, value=value, constant_result=value)
            if unode.value:
                values.append(unode)
        else:
            values.extend(substrings)
    if not values:
        value = EncodedString('')
        node = ExprNodes.UnicodeNode(node.pos, value=value, constant_result=value)
    elif len(values) == 1:
        node = values[0]
    elif len(values) == 2:
        node = ExprNodes.binop_node(node.pos, '+', *values)
    else:
        node.values = values
    return node