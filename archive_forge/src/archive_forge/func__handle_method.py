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
def _handle_method(self, node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs):
    """
        Try to inject C-API calls for unbound method calls to builtin types.
        While the method declarations in Builtin.py already handle this, we
        can additionally resolve bound and unbound methods here that were
        assigned to variables ahead of time.
        """
    if kwargs:
        return node
    if not function or not function.is_attribute or (not function.obj.is_name):
        return node
    type_entry = self.current_env().lookup(type_name)
    if not type_entry:
        return node
    method = ExprNodes.AttributeNode(node.function.pos, obj=ExprNodes.NameNode(function.pos, name=type_name, entry=type_entry, type=type_entry.type), attribute=attr_name, is_called=True).analyse_as_type_attribute(self.current_env())
    if method is None:
        return self._optimise_generic_builtin_method_call(node, attr_name, function, arg_list, is_unbound_method)
    args = node.args
    if args is None and node.arg_tuple:
        args = node.arg_tuple.args
    call_node = ExprNodes.SimpleCallNode(node.pos, function=method, args=args)
    if not is_unbound_method:
        call_node.self = function.obj
    call_node.analyse_c_function_call(self.current_env())
    call_node.analysed = True
    return call_node.coerce_to(node.type, self.current_env())