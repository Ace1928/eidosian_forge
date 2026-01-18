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
def visit_ModNode(self, node):
    self.visitchildren(node)
    if isinstance(node.operand1, ExprNodes.UnicodeNode) and isinstance(node.operand2, ExprNodes.TupleNode):
        if not node.operand2.mult_factor:
            fstring = self._build_fstring(node.operand1.pos, node.operand1.value, node.operand2.args)
            if fstring is not None:
                return fstring
    return self.visit_BinopNode(node)