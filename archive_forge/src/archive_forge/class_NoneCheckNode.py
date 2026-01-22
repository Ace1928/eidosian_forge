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
class NoneCheckNode(CoercionNode):
    is_nonecheck = True

    def __init__(self, arg, exception_type_cname, exception_message, exception_format_args=()):
        CoercionNode.__init__(self, arg)
        self.type = arg.type
        self.result_ctype = arg.ctype()
        self.exception_type_cname = exception_type_cname
        self.exception_message = exception_message
        self.exception_format_args = tuple(exception_format_args or ())
    nogil_check = None

    def analyse_types(self, env):
        return self

    def may_be_none(self):
        return False

    def is_simple(self):
        return self.arg.is_simple()

    def result_in_temp(self):
        return self.arg.result_in_temp()

    def nonlocally_immutable(self):
        return self.arg.nonlocally_immutable()

    def calculate_result_code(self):
        return self.arg.result()

    def condition(self):
        if self.type.is_pyobject:
            return self.arg.py_result()
        elif self.type.is_memoryviewslice:
            return '((PyObject *) %s.memview)' % self.arg.result()
        else:
            raise Exception('unsupported type')

    @classmethod
    def generate(cls, arg, code, exception_message, exception_type_cname='PyExc_TypeError', exception_format_args=(), in_nogil_context=False):
        node = cls(arg, exception_type_cname, exception_message, exception_format_args)
        node.in_nogil_context = in_nogil_context
        node.put_nonecheck(code)

    @classmethod
    def generate_if_needed(cls, arg, code, exception_message, exception_type_cname='PyExc_TypeError', exception_format_args=(), in_nogil_context=False):
        if arg.may_be_none():
            cls.generate(arg, code, exception_message, exception_type_cname, exception_format_args, in_nogil_context)

    def put_nonecheck(self, code):
        code.putln('if (unlikely(%s == Py_None)) {' % self.condition())
        if self.in_nogil_context:
            code.put_ensure_gil()
        escape = StringEncoding.escape_byte_string
        if self.exception_format_args:
            code.putln('PyErr_Format(%s, "%s", %s);' % (self.exception_type_cname, StringEncoding.escape_byte_string(self.exception_message.encode('UTF-8')), ', '.join(['"%s"' % escape(str(arg).encode('UTF-8')) for arg in self.exception_format_args])))
        else:
            code.putln('PyErr_SetString(%s, "%s");' % (self.exception_type_cname, escape(self.exception_message.encode('UTF-8'))))
        if self.in_nogil_context:
            code.put_release_ensured_gil()
        code.putln(code.error_goto(self.pos))
        code.putln('}')

    def generate_result_code(self, code):
        self.put_nonecheck(code)

    def generate_post_assignment_code(self, code):
        self.arg.generate_post_assignment_code(code)

    def free_temps(self, code):
        self.arg.free_temps(code)