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
def visit_CoerceToPyTypeNode(self, node):
    """Drop redundant conversion nodes after tree changes."""
    self.visitchildren(node)
    arg = node.arg
    if isinstance(arg, ExprNodes.CoerceFromPyTypeNode):
        arg = arg.arg
    if isinstance(arg, ExprNodes.PythonCapiCallNode):
        if arg.function.name == 'float' and len(arg.args) == 1:
            func_arg = arg.args[0]
            if func_arg.type is Builtin.float_type:
                return func_arg.as_none_safe_node("float() argument must be a string or a number, not 'NoneType'")
            elif func_arg.type.is_pyobject and arg.function.cname == '__Pyx_PyObject_AsDouble':
                return ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyNumber_Float', self.PyNumber_Float_func_type, args=[func_arg], py_name='float', is_temp=node.is_temp, utility_code=UtilityCode.load_cached('pynumber_float', 'TypeConversion.c'), result_is_used=node.result_is_used).coerce_to(node.type, self.current_env())
    return node