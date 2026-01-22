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
def _wrap_self_arg(self, self_arg, function, is_unbound_method, attr_name):
    if self_arg.is_literal:
        return self_arg
    if is_unbound_method:
        self_arg = self_arg.as_none_safe_node("descriptor '%s' requires a '%s' object but received a 'NoneType'", format_args=[attr_name, self_arg.type.name])
    else:
        self_arg = self_arg.as_none_safe_node("'NoneType' object has no attribute '%{0}s'".format('.30' if len(attr_name) <= 30 else ''), error='PyExc_AttributeError', format_args=[attr_name])
    return self_arg