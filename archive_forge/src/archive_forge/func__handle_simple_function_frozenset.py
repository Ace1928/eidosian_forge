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
def _handle_simple_function_frozenset(self, node, function, pos_args):
    if not pos_args:
        pos_args = [ExprNodes.NullNode(node.pos)]
    elif len(pos_args) > 1:
        return node
    elif pos_args[0].type is Builtin.frozenset_type and (not pos_args[0].may_be_none()):
        return pos_args[0]
    return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyFrozenSet_New', self.PyFrozenSet_New_func_type, args=pos_args, is_temp=node.is_temp, utility_code=UtilityCode.load_cached('pyfrozenset_new', 'Builtins.c'), py_name='frozenset')