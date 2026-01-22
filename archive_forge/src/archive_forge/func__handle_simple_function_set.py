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
def _handle_simple_function_set(self, node, function, pos_args):
    if len(pos_args) != 1:
        return node
    if pos_args[0].is_sequence_constructor:
        args = []
        temps = []
        for arg in pos_args[0].args:
            if not arg.is_simple():
                arg = UtilNodes.LetRefNode(arg)
                temps.append(arg)
            args.append(arg)
        result = ExprNodes.SetNode(node.pos, is_temp=1, args=args)
        self.replace(node, result)
        for temp in temps[::-1]:
            result = UtilNodes.EvalWithTempExprNode(temp, result)
        return result
    else:
        return self.replace(node, ExprNodes.PythonCapiCallNode(node.pos, 'PySet_New', self.PySet_New_func_type, args=pos_args, is_temp=node.is_temp, py_name='set'))