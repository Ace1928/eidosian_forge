from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class ForFromStatNode(LoopNode, StatNode):
    child_attrs = ['target', 'bound1', 'bound2', 'step', 'body', 'else_clause']
    is_py_target = False
    loopvar_node = None
    py_loopvar_node = None
    from_range = False
    gil_message = 'For-loop using object bounds or target'

    def nogil_check(self, env):
        for x in (self.target, self.bound1, self.bound2):
            if x.type.is_pyobject:
                self.gil_error()

    def analyse_declarations(self, env):
        self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.target = self.target.analyse_target_types(env)
        self.bound1 = self.bound1.analyse_types(env)
        self.bound2 = self.bound2.analyse_types(env)
        if self.step is not None:
            if isinstance(self.step, ExprNodes.UnaryMinusNode):
                warning(self.step.pos, 'Probable infinite loop in for-from-by statement. Consider switching the directions of the relations.', 2)
            self.step = self.step.analyse_types(env)
        self.set_up_loop(env)
        target_type = self.target.type
        if not (target_type.is_pyobject or target_type.is_numeric):
            error(self.target.pos, 'for-from loop variable must be c numeric type or Python object')
        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def set_up_loop(self, env):
        from . import ExprNodes
        target_type = self.target.type
        if target_type.is_numeric:
            loop_type = target_type
        else:
            if target_type.is_enum:
                warning(self.target.pos, 'Integer loops over enum values are fragile. Please cast to a safe integer type instead.')
            loop_type = PyrexTypes.c_long_type if target_type.is_pyobject else PyrexTypes.c_int_type
            if not self.bound1.type.is_pyobject:
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound1.type)
            if not self.bound2.type.is_pyobject:
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound2.type)
            if self.step is not None and (not self.step.type.is_pyobject):
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.step.type)
        self.bound1 = self.bound1.coerce_to(loop_type, env)
        self.bound2 = self.bound2.coerce_to(loop_type, env)
        if not self.bound2.is_literal:
            self.bound2 = self.bound2.coerce_to_temp(env)
        if self.step is not None:
            self.step = self.step.coerce_to(loop_type, env)
            if not self.step.is_literal:
                self.step = self.step.coerce_to_temp(env)
        if target_type.is_numeric or target_type.is_enum:
            self.is_py_target = False
            if isinstance(self.target, ExprNodes.BufferIndexNode):
                raise error(self.pos, 'Buffer or memoryview slicing/indexing not allowed as for-loop target.')
            self.loopvar_node = self.target
            self.py_loopvar_node = None
        else:
            self.is_py_target = True
            c_loopvar_node = ExprNodes.TempNode(self.pos, loop_type, env)
            self.loopvar_node = c_loopvar_node
            self.py_loopvar_node = ExprNodes.CloneNode(c_loopvar_node).coerce_to_pyobject(env)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        from_range = self.from_range
        self.bound1.generate_evaluation_code(code)
        self.bound2.generate_evaluation_code(code)
        offset, incop = self.relation_table[self.relation1]
        if self.step is not None:
            self.step.generate_evaluation_code(code)
            step = self.step.result()
            incop = '%s=%s' % (incop[0], step)
        else:
            step = '1'
        from . import ExprNodes
        if isinstance(self.loopvar_node, ExprNodes.TempNode):
            self.loopvar_node.allocate(code)
        if isinstance(self.py_loopvar_node, ExprNodes.TempNode):
            self.py_loopvar_node.allocate(code)
        loopvar_type = PyrexTypes.c_long_type if self.target.type.is_enum else self.target.type
        if from_range and (not self.is_py_target):
            loopvar_name = code.funcstate.allocate_temp(loopvar_type, False)
        else:
            loopvar_name = self.loopvar_node.result()
        if loopvar_type.is_int and (not loopvar_type.signed) and (self.relation2[0] == '>'):
            code.putln('for (%s = %s%s + %s; %s %s %s + %s; ) { %s%s;' % (loopvar_name, self.bound1.result(), offset, step, loopvar_name, self.relation2, self.bound2.result(), step, loopvar_name, incop))
        else:
            code.putln('for (%s = %s%s; %s %s %s; %s%s) {' % (loopvar_name, self.bound1.result(), offset, loopvar_name, self.relation2, self.bound2.result(), loopvar_name, incop))
        coerced_loopvar_node = self.py_loopvar_node
        if coerced_loopvar_node is None and from_range:
            coerced_loopvar_node = ExprNodes.RawCNameExprNode(self.target.pos, loopvar_type, loopvar_name)
        if coerced_loopvar_node is not None:
            coerced_loopvar_node.generate_evaluation_code(code)
            self.target.generate_assignment_code(coerced_loopvar_node, code)
        self.body.generate_execution_code(code)
        code.put_label(code.continue_label)
        if not from_range and self.py_loopvar_node:
            if self.target.entry.is_pyglobal:
                target_node = ExprNodes.PyTempNode(self.target.pos, None)
                target_node.allocate(code)
                interned_cname = code.intern_identifier(self.target.entry.name)
                if self.target.entry.scope.is_module_scope:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('GetModuleGlobalName', 'ObjectHandling.c'))
                    lookup_func = '__Pyx_GetModuleGlobalName(%s, %s); %s'
                else:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('GetNameInClass', 'ObjectHandling.c'))
                    lookup_func = '__Pyx_GetNameInClass(%s, {}, %s); %s'.format(self.target.entry.scope.namespace_cname)
                code.putln(lookup_func % (target_node.result(), interned_cname, code.error_goto_if_null(target_node.result(), self.target.pos)))
                target_node.generate_gotref(code)
            else:
                target_node = self.target
            from_py_node = ExprNodes.CoerceFromPyTypeNode(self.loopvar_node.type, target_node, self.target.entry.scope)
            from_py_node.temp_code = loopvar_name
            from_py_node.generate_result_code(code)
            if self.target.entry.is_pyglobal:
                code.put_decref(target_node.result(), target_node.type)
                target_node.release(code)
        code.putln('}')
        if not from_range and self.py_loopvar_node:
            self.py_loopvar_node.generate_evaluation_code(code)
            self.target.generate_assignment_code(self.py_loopvar_node, code)
        if from_range and (not self.is_py_target):
            code.funcstate.release_temp(loopvar_name)
        break_label = code.break_label
        code.set_loop_labels(old_loop_labels)
        if self.else_clause:
            code.putln('/*else*/ {')
            self.else_clause.generate_execution_code(code)
            code.putln('}')
        code.put_label(break_label)
        self.bound1.generate_disposal_code(code)
        self.bound1.free_temps(code)
        self.bound2.generate_disposal_code(code)
        self.bound2.free_temps(code)
        if isinstance(self.loopvar_node, ExprNodes.TempNode):
            self.loopvar_node.release(code)
        if isinstance(self.py_loopvar_node, ExprNodes.TempNode):
            self.py_loopvar_node.release(code)
        if self.step is not None:
            self.step.generate_disposal_code(code)
            self.step.free_temps(code)
    relation_table = {'<=': ('', '++'), '<': ('+1', '++'), '>=': ('', '--'), '>': ('-1', '--')}

    def generate_function_definitions(self, env, code):
        self.target.generate_function_definitions(env, code)
        self.bound1.generate_function_definitions(env, code)
        self.bound2.generate_function_definitions(env, code)
        if self.step is not None:
            self.step.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.target.annotate(code)
        self.bound1.annotate(code)
        self.bound2.annotate(code)
        if self.step:
            self.step.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)