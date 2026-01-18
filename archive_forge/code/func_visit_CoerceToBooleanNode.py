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
def visit_CoerceToBooleanNode(self, node):
    """Drop redundant conversion nodes after tree changes.
        """
    self.visitchildren(node)
    arg = node.arg
    if isinstance(arg, ExprNodes.PyTypeTestNode):
        arg = arg.arg
    if isinstance(arg, ExprNodes.CoerceToPyTypeNode):
        if arg.type in (PyrexTypes.py_object_type, Builtin.bool_type):
            return arg.arg.coerce_to_boolean(self.current_env())
    return node