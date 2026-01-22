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
def _handle_simple_function_isinstance(self, node, function, pos_args):
    """Replace isinstance() checks against builtin types by the
        corresponding C-API call.
        """
    if len(pos_args) != 2:
        return node
    arg, types = pos_args
    temps = []
    if isinstance(types, ExprNodes.TupleNode):
        types = types.args
        if len(types) == 1 and (not types[0].type is Builtin.type_type):
            return node
        if arg.is_attribute or not arg.is_simple():
            arg = UtilNodes.ResultRefNode(arg)
            temps.append(arg)
    elif types.type is Builtin.type_type:
        types = [types]
    else:
        return node
    tests = []
    test_nodes = []
    env = self.current_env()
    for test_type_node in types:
        builtin_type = None
        if test_type_node.is_name:
            if test_type_node.entry:
                entry = env.lookup(test_type_node.entry.name)
                if entry and entry.type and entry.type.is_builtin_type:
                    builtin_type = entry.type
        if builtin_type is Builtin.type_type:
            if entry.name != 'type' or not (entry.scope and entry.scope.is_builtin_scope):
                builtin_type = None
        if builtin_type is not None:
            type_check_function = entry.type.type_check_function(exact=False)
            if type_check_function == '__Pyx_Py3Int_Check' and builtin_type is Builtin.int_type:
                type_check_function = 'PyInt_Check'
            if type_check_function in tests:
                continue
            tests.append(type_check_function)
            type_check_args = [arg]
        elif test_type_node.type is Builtin.type_type:
            type_check_function = '__Pyx_TypeCheck'
            type_check_args = [arg, test_type_node]
        else:
            if not test_type_node.is_literal:
                test_type_node = UtilNodes.ResultRefNode(test_type_node)
                temps.append(test_type_node)
            type_check_function = 'PyObject_IsInstance'
            type_check_args = [arg, test_type_node]
        test_nodes.append(ExprNodes.PythonCapiCallNode(test_type_node.pos, type_check_function, self.Py_type_check_func_type, args=type_check_args, is_temp=True))

    def join_with_or(a, b, make_binop_node=ExprNodes.binop_node):
        or_node = make_binop_node(node.pos, 'or', a, b)
        or_node.type = PyrexTypes.c_bint_type
        or_node.wrap_operands(env)
        return or_node
    test_node = reduce(join_with_or, test_nodes).coerce_to(node.type, env)
    for temp in temps[::-1]:
        test_node = UtilNodes.EvalWithTempExprNode(temp, test_node)
    return test_node