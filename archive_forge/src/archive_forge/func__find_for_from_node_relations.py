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
def _find_for_from_node_relations(self, neg_step_value, reversed):
    if reversed:
        if neg_step_value:
            return ('<', '<=')
        else:
            return ('>', '>=')
    elif neg_step_value:
        return ('>=', '>')
    else:
        return ('<=', '<')