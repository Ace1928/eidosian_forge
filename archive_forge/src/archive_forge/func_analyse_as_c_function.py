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
def analyse_as_c_function(self, env):
    base_type = self.base.type
    if base_type.is_fused:
        self.parse_indexed_fused_cdef(env)
    else:
        self.type_indices = self.parse_index_as_types(env)
        self.index = None
        if base_type.templates is None:
            error(self.pos, 'Can only parameterize template functions.')
            self.type = error_type
        elif self.type_indices is None:
            self.type = error_type
        elif len(base_type.templates) != len(self.type_indices):
            error(self.pos, 'Wrong number of template arguments: expected %s, got %s' % (len(base_type.templates), len(self.type_indices)))
            self.type = error_type
        else:
            self.type = base_type.specialize(dict(zip(base_type.templates, self.type_indices)))
    return self