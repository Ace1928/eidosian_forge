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
class ScopedExprNode(ExprNode):
    subexprs = []
    expr_scope = None
    has_local_scope = True

    def init_scope(self, outer_scope, expr_scope=None):
        if expr_scope is not None:
            self.expr_scope = expr_scope
        elif self.has_local_scope:
            self.expr_scope = Symtab.ComprehensionScope(outer_scope)
        elif not self.expr_scope:
            self.expr_scope = None

    def analyse_declarations(self, env):
        self.init_scope(env)

    def analyse_scoped_declarations(self, env):
        pass

    def analyse_types(self, env):
        return self

    def analyse_scoped_expressions(self, env):
        return self

    def generate_evaluation_code(self, code):
        generate_inner_evaluation_code = super(ScopedExprNode, self).generate_evaluation_code
        if not self.has_local_scope or not self.expr_scope.var_entries:
            generate_inner_evaluation_code(code)
            return
        code.putln('{ /* enter inner scope */')
        py_entries = []
        for _, entry in sorted((item for item in self.expr_scope.entries.items() if item[0])):
            if not entry.in_closure:
                if entry.type.is_pyobject and entry.used:
                    py_entries.append(entry)
        if not py_entries:
            generate_inner_evaluation_code(code)
            code.putln('} /* exit inner scope */')
            return
        old_loop_labels = code.new_loop_labels()
        old_error_label = code.new_error_label()
        generate_inner_evaluation_code(code)
        self._generate_vars_cleanup(code, py_entries)
        exit_scope = code.new_label('exit_scope')
        code.put_goto(exit_scope)
        for label, old_label in [(code.error_label, old_error_label)] + list(zip(code.get_loop_labels(), old_loop_labels)):
            if code.label_used(label):
                code.put_label(label)
                self._generate_vars_cleanup(code, py_entries)
                code.put_goto(old_label)
        code.put_label(exit_scope)
        code.putln('} /* exit inner scope */')
        code.set_loop_labels(old_loop_labels)
        code.error_label = old_error_label

    def _generate_vars_cleanup(self, code, py_entries):
        for entry in py_entries:
            if entry.is_cglobal:
                code.put_var_gotref(entry)
                code.put_var_decref_set(entry, 'Py_None')
            else:
                code.put_var_xdecref_clear(entry)