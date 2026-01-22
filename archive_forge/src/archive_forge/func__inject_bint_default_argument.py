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
def _inject_bint_default_argument(self, node, args, arg_index, default_value):
    assert len(args) >= arg_index
    if len(args) == arg_index:
        default_value = bool(default_value)
        args.append(ExprNodes.BoolNode(node.pos, value=default_value, constant_result=default_value))
    else:
        args[arg_index] = args[arg_index].coerce_to_boolean(self.current_env())