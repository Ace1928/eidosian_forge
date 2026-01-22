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
def _handle_simple_function_unicode(self, node, function, pos_args):
    """Optimise single argument calls to unicode().
        """
    if len(pos_args) != 1:
        if len(pos_args) == 0:
            return ExprNodes.UnicodeNode(node.pos, value=EncodedString(), constant_result=u'')
        return node
    arg = pos_args[0]
    if arg.type is Builtin.unicode_type:
        if not arg.may_be_none():
            return arg
        cname = '__Pyx_PyUnicode_Unicode'
        utility_code = UtilityCode.load_cached('PyUnicode_Unicode', 'StringTools.c')
    else:
        cname = '__Pyx_PyObject_Unicode'
        utility_code = UtilityCode.load_cached('PyObject_Unicode', 'StringTools.c')
    return ExprNodes.PythonCapiCallNode(node.pos, cname, self.PyObject_Unicode_func_type, args=pos_args, is_temp=node.is_temp, utility_code=utility_code, py_name='unicode')