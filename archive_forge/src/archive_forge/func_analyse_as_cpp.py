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
def analyse_as_cpp(self, env, setting):
    base_type = self.base.type
    function = env.lookup_operator('[]', [self.base, self.index])
    if function is None:
        error(self.pos, "Indexing '%s' not supported for index type '%s'" % (base_type, self.index.type))
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'
        return self
    func_type = function.type
    if func_type.is_ptr:
        func_type = func_type.base_type
    self.exception_check = func_type.exception_check
    self.exception_value = func_type.exception_value
    if self.exception_check:
        if not setting:
            self.is_temp = True
        if needs_cpp_exception_conversion(self):
            env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
    self.index = self.index.coerce_to(func_type.args[0].type, env)
    self.type = func_type.return_type
    if setting and (not func_type.return_type.is_reference):
        error(self.pos, "Can't set non-reference result '%s'" % self.type)
    return self