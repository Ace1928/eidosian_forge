from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def analyse_as_pyobject(self, env, is_slice, getting, setting):
    base_type = self.base.type
    if self.index.type.is_unicode_char and base_type is not dict_type:
        warning(self.pos, 'Item lookup of unicode character codes now always converts to a Unicode string. Use an explicit C integer cast to get back the previous integer lookup behaviour.', level=1)
        self.index = self.index.coerce_to_pyobject(env)
        self.is_temp = 1
    elif self.index.type.is_int and base_type is not dict_type:
        if getting and (not env.directives['boundscheck']) and (base_type in (list_type, tuple_type, bytearray_type)) and (not self.index.type.signed or not env.directives['wraparound'] or (isinstance(self.index, IntNode) and self.index.has_constant_result() and (self.index.constant_result >= 0))):
            self.is_temp = 0
        else:
            self.is_temp = 1
        self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env).coerce_to_simple(env)
        self.original_index_type.create_to_py_utility_code(env)
    else:
        self.index = self.index.coerce_to_pyobject(env)
        self.is_temp = 1
    if self.index.type.is_int and base_type is unicode_type:
        self.type = PyrexTypes.c_py_ucs4_type
    elif self.index.type.is_int and base_type is bytearray_type:
        if setting:
            self.type = PyrexTypes.c_uchar_type
        else:
            self.type = PyrexTypes.c_int_type
    elif is_slice and base_type in (bytes_type, bytearray_type, str_type, unicode_type, list_type, tuple_type):
        self.type = base_type
    else:
        item_type = None
        if base_type in (list_type, tuple_type) and self.index.type.is_int:
            item_type = infer_sequence_item_type(env, self.base, self.index, seq_type=base_type)
        if base_type in (list_type, tuple_type, dict_type):
            self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")
        if item_type is None or not item_type.is_pyobject:
            self.type = py_object_type
        else:
            self.type = item_type
    self.wrap_in_nonecheck_node(env, getting)
    return self