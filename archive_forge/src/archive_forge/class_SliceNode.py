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
class SliceNode(ExprNode):
    subexprs = ['start', 'stop', 'step']
    is_slice = True
    type = slice_type
    is_temp = 1

    def calculate_constant_result(self):
        self.constant_result = slice(self.start.constant_result, self.stop.constant_result, self.step.constant_result)

    def compile_time_value(self, denv):
        start = self.start.compile_time_value(denv)
        stop = self.stop.compile_time_value(denv)
        step = self.step.compile_time_value(denv)
        try:
            return slice(start, stop, step)
        except Exception as e:
            self.compile_time_value_error(e)

    def may_be_none(self):
        return False

    def analyse_types(self, env):
        start = self.start.analyse_types(env)
        stop = self.stop.analyse_types(env)
        step = self.step.analyse_types(env)
        self.start = start.coerce_to_pyobject(env)
        self.stop = stop.coerce_to_pyobject(env)
        self.step = step.coerce_to_pyobject(env)
        if self.start.is_literal and self.stop.is_literal and self.step.is_literal:
            self.is_literal = True
            self.is_temp = False
        return self
    gil_message = 'Constructing Python slice object'

    def calculate_result_code(self):
        return self.result_code

    def generate_result_code(self, code):
        if self.is_literal:
            dedup_key = make_dedup_key(self.type, (self,))
            self.result_code = code.get_py_const(py_object_type, 'slice', cleanup_level=2, dedup_key=dedup_key)
            code = code.get_cached_constants_writer(self.result_code)
            if code is None:
                return
            code.mark_pos(self.pos)
        code.putln('%s = PySlice_New(%s, %s, %s); %s' % (self.result(), self.start.py_result(), self.stop.py_result(), self.step.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)
        if self.is_literal:
            self.generate_giveref(code)