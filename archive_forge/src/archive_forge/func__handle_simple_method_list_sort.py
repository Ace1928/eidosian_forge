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
def _handle_simple_method_list_sort(self, node, function, args, is_unbound_method):
    """Call PyList_Sort() instead of the 0-argument l.sort().
        """
    if len(args) != 1:
        return node
    return self._substitute_method_call(node, function, 'PyList_Sort', self.single_param_func_type, 'sort', is_unbound_method, args).coerce_to(node.type, self.current_env)