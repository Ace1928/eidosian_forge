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
def _handle_simple_function_bool(self, node, function, pos_args):
    """Transform bool(x) into a type coercion to a boolean.
        """
    if len(pos_args) == 0:
        return ExprNodes.BoolNode(node.pos, value=False, constant_result=False).coerce_to(Builtin.bool_type, self.current_env())
    elif len(pos_args) != 1:
        self._error_wrong_arg_count('bool', node, pos_args, '0 or 1')
        return node
    else:
        operand = pos_args[0].coerce_to_boolean(self.current_env())
        operand = ExprNodes.NotNode(node.pos, operand=operand)
        operand = ExprNodes.NotNode(node.pos, operand=operand)
        return operand.coerce_to_pyobject(self.current_env())