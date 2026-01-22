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
def _optimise_num_binop(self, operator, node, function, args, is_unbound_method):
    """
        Optimise math operators for (likely) float or small integer operations.
        """
    if getattr(node, 'special_bool_cmp_function', None):
        return node
    if len(args) != 2:
        return node
    if node.type.is_pyobject:
        ret_type = PyrexTypes.py_object_type
    elif node.type is PyrexTypes.c_bint_type and operator in ('Eq', 'Ne'):
        ret_type = PyrexTypes.c_bint_type
    else:
        return node
    result = optimise_numeric_binop(operator, node, ret_type, args[0], args[1])
    if not result:
        return node
    func_cname, utility_code, extra_args, num_type = result
    args = list(args) + extra_args
    call_node = self._substitute_method_call(node, function, func_cname, self.Pyx_BinopInt_func_types[num_type, ret_type], '__%s__' % operator[:3].lower(), is_unbound_method, args, may_return_none=True, with_none_check=False, utility_code=utility_code)
    if node.type.is_pyobject and (not ret_type.is_pyobject):
        call_node = ExprNodes.CoerceToPyTypeNode(call_node, self.current_env(), node.type)
    return call_node