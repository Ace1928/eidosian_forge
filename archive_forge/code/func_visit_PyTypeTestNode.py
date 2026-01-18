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
def visit_PyTypeTestNode(self, node):
    """Remove tests for alternatively allowed None values from
        type tests when we know that the argument cannot be None
        anyway.
        """
    self.visitchildren(node)
    if not node.notnone:
        if not node.arg.may_be_none():
            node.notnone = True
    return node