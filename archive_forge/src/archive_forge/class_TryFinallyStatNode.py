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
class TryFinallyStatNode(StatNode):
    child_attrs = ['body', 'finally_clause', 'finally_except_clause']
    preserve_exception = 1
    handle_error_case = True
    func_return_type = None
    finally_except_clause = None
    is_try_finally_in_nogil = False
    in_generator = False

    @staticmethod
    def create_analysed(pos, env, body, finally_clause):
        node = TryFinallyStatNode(pos, body=body, finally_clause=finally_clause)
        return node

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        self.finally_except_clause = copy.deepcopy(self.finally_clause)
        self.finally_except_clause.analyse_declarations(env)
        self.finally_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        self.finally_clause = self.finally_clause.analyse_expressions(env)
        self.finally_except_clause = self.finally_except_clause.analyse_expressions(env)
        if env.return_type and (not env.return_type.is_void):
            self.func_return_type = env.return_type
        return self
    nogil_check = Node.gil_error
    gil_message = 'Try-finally statement'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.putln('/*try:*/ {')
        old_error_label = code.error_label
        old_labels = code.all_new_labels()
        new_labels = code.get_all_labels()
        new_error_label = code.error_label
        if not self.handle_error_case:
            code.error_label = old_error_label
        catch_label = code.new_label()
        was_in_try_finally = code.funcstate.in_try_finally
        code.funcstate.in_try_finally = 1
        self.body.generate_execution_code(code)
        code.funcstate.in_try_finally = was_in_try_finally
        code.putln('}')
        temps_to_clean_up = code.funcstate.all_free_managed_temps()
        code.mark_pos(self.finally_clause.pos)
        code.putln('/*finally:*/ {')
        code.set_all_labels(old_labels)

        def fresh_finally_clause(_next=[self.finally_clause]):
            node = _next[0]
            node_copy = copy.deepcopy(node)
            if node is self.finally_clause:
                _next[0] = node_copy
            else:
                node = node_copy
            return node
        preserve_error = self.preserve_exception and code.label_used(new_error_label)
        needs_success_cleanup = not self.finally_clause.is_terminator
        if not self.body.is_terminator:
            code.putln('/*normal exit:*/{')
            fresh_finally_clause().generate_execution_code(code)
            if not self.finally_clause.is_terminator:
                code.put_goto(catch_label)
            code.putln('}')
        if preserve_error:
            code.put_label(new_error_label)
            code.putln('/*exception exit:*/{')
            if not self.in_generator:
                code.putln('__Pyx_PyThreadState_declare')
            if self.is_try_finally_in_nogil:
                code.declare_gilstate()
            if needs_success_cleanup:
                exc_lineno_cnames = tuple([code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False) for _ in range(2)])
                exc_filename_cname = code.funcstate.allocate_temp(PyrexTypes.CPtrType(PyrexTypes.c_const_type(PyrexTypes.c_char_type)), manage_ref=False)
            else:
                exc_lineno_cnames = exc_filename_cname = None
            exc_vars = tuple([code.funcstate.allocate_temp(py_object_type, manage_ref=False) for _ in range(6)])
            self.put_error_catcher(code, temps_to_clean_up, exc_vars, exc_lineno_cnames, exc_filename_cname)
            finally_old_labels = code.all_new_labels()
            code.putln('{')
            old_exc_vars = code.funcstate.exc_vars
            code.funcstate.exc_vars = exc_vars[:3]
            self.finally_except_clause.generate_execution_code(code)
            code.funcstate.exc_vars = old_exc_vars
            code.putln('}')
            if needs_success_cleanup:
                self.put_error_uncatcher(code, exc_vars, exc_lineno_cnames, exc_filename_cname)
                if exc_lineno_cnames:
                    for cname in exc_lineno_cnames:
                        code.funcstate.release_temp(cname)
                if exc_filename_cname:
                    code.funcstate.release_temp(exc_filename_cname)
                code.put_goto(old_error_label)
            for _ in code.label_interceptor(code.get_all_labels(), finally_old_labels):
                self.put_error_cleaner(code, exc_vars)
            for cname in exc_vars:
                code.funcstate.release_temp(cname)
            code.putln('}')
        code.set_all_labels(old_labels)
        return_label = code.return_label
        exc_vars = ()
        for i, (new_label, old_label) in enumerate(zip(new_labels, old_labels)):
            if not code.label_used(new_label):
                continue
            if new_label == new_error_label and preserve_error:
                continue
            code.putln('%s: {' % new_label)
            ret_temp = None
            if old_label == return_label:
                if self.in_generator:
                    exc_vars = tuple([code.funcstate.allocate_temp(py_object_type, manage_ref=False) for _ in range(6)])
                    self.put_error_catcher(code, [], exc_vars)
                if not self.finally_clause.is_terminator:
                    if self.func_return_type and (not self.is_try_finally_in_nogil) and (not isinstance(self.finally_clause, GILExitNode)):
                        ret_temp = code.funcstate.allocate_temp(self.func_return_type, manage_ref=False)
                        code.putln('%s = %s;' % (ret_temp, Naming.retval_cname))
                        if self.func_return_type.is_pyobject:
                            code.putln('%s = 0;' % Naming.retval_cname)
            fresh_finally_clause().generate_execution_code(code)
            if old_label == return_label:
                if ret_temp:
                    code.putln('%s = %s;' % (Naming.retval_cname, ret_temp))
                    if self.func_return_type.is_pyobject:
                        code.putln('%s = 0;' % ret_temp)
                    code.funcstate.release_temp(ret_temp)
                if self.in_generator:
                    self.put_error_uncatcher(code, exc_vars)
                    for cname in exc_vars:
                        code.funcstate.release_temp(cname)
            if not self.finally_clause.is_terminator:
                code.put_goto(old_label)
            code.putln('}')
        code.put_label(catch_label)
        code.putln('}')

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)
        self.finally_clause.generate_function_definitions(env, code)
        if self.finally_except_clause:
            self.finally_except_clause.generate_function_definitions(env, code)

    def put_error_catcher(self, code, temps_to_clean_up, exc_vars, exc_lineno_cnames=None, exc_filename_cname=None):
        code.globalstate.use_utility_code(restore_exception_utility_code)
        code.globalstate.use_utility_code(get_exception_utility_code)
        code.globalstate.use_utility_code(swap_exception_utility_code)
        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)
        code.putln('__Pyx_PyThreadState_assign')
        code.putln(' '.join(['%s = 0;' % var for var in exc_vars]))
        for temp_name, type in temps_to_clean_up:
            code.put_xdecref_clear(temp_name, type)
        code.putln('if (PY_MAJOR_VERSION >= 3) __Pyx_ExceptionSwap(&%s, &%s, &%s);' % exc_vars[3:])
        code.putln('if ((PY_MAJOR_VERSION < 3) || unlikely(__Pyx_GetException(&%s, &%s, &%s) < 0)) __Pyx_ErrFetch(&%s, &%s, &%s);' % (exc_vars[:3] * 2))
        for var in exc_vars:
            code.put_xgotref(var, py_object_type)
        if exc_lineno_cnames:
            code.putln('%s = %s; %s = %s; %s = %s;' % (exc_lineno_cnames[0], Naming.lineno_cname, exc_lineno_cnames[1], Naming.clineno_cname, exc_filename_cname, Naming.filename_cname))
        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()

    def put_error_uncatcher(self, code, exc_vars, exc_lineno_cnames=None, exc_filename_cname=None):
        code.globalstate.use_utility_code(restore_exception_utility_code)
        code.globalstate.use_utility_code(reset_exception_utility_code)
        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)
            code.putln('__Pyx_PyThreadState_assign')
        code.putln('if (PY_MAJOR_VERSION >= 3) {')
        for var in exc_vars[3:]:
            code.put_xgiveref(var, py_object_type)
        code.putln('__Pyx_ExceptionReset(%s, %s, %s);' % exc_vars[3:])
        code.putln('}')
        for var in exc_vars[:3]:
            code.put_xgiveref(var, py_object_type)
        code.putln('__Pyx_ErrRestore(%s, %s, %s);' % exc_vars[:3])
        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()
        code.putln(' '.join(['%s = 0;' % var for var in exc_vars]))
        if exc_lineno_cnames:
            code.putln('%s = %s; %s = %s; %s = %s;' % (Naming.lineno_cname, exc_lineno_cnames[0], Naming.clineno_cname, exc_lineno_cnames[1], Naming.filename_cname, exc_filename_cname))

    def put_error_cleaner(self, code, exc_vars):
        code.globalstate.use_utility_code(reset_exception_utility_code)
        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)
            code.putln('__Pyx_PyThreadState_assign')
        code.putln('if (PY_MAJOR_VERSION >= 3) {')
        for var in exc_vars[3:]:
            code.put_xgiveref(var, py_object_type)
        code.putln('__Pyx_ExceptionReset(%s, %s, %s);' % exc_vars[3:])
        code.putln('}')
        for var in exc_vars[:3]:
            code.put_xdecref_clear(var, py_object_type)
        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()
        code.putln(' '.join(['%s = 0;'] * 3) % exc_vars[3:])

    def annotate(self, code):
        self.body.annotate(code)
        self.finally_clause.annotate(code)