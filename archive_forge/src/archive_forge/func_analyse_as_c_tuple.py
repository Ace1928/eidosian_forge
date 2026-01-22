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
def analyse_as_c_tuple(self, env, getting, setting):
    base_type = self.base.type
    if isinstance(self.index, IntNode) and self.index.has_constant_result():
        index = self.index.constant_result
        if -base_type.size <= index < base_type.size:
            if index < 0:
                index += base_type.size
            self.type = base_type.components[index]
        else:
            error(self.pos, "Index %s out of bounds for '%s'" % (index, base_type))
            self.type = PyrexTypes.error_type
        return self
    else:
        self.base = self.base.coerce_to_pyobject(env)
        return self.analyse_base_and_index_types(env, getting=getting, setting=setting, analyse_base=False)