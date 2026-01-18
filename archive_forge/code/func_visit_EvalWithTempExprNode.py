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
def visit_EvalWithTempExprNode(self, node):
    if not self.current_directives.get('optimize.use_switch'):
        self.visitchildren(node)
        return node
    orig_expr = node.subexpression
    temp_ref = node.lazy_temp
    self.visitchildren(node)
    if node.subexpression is not orig_expr:
        if not Visitor.tree_contains(node.subexpression, temp_ref):
            return node.subexpression
    return node