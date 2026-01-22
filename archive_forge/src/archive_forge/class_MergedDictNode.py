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
class MergedDictNode(ExprNode):
    subexprs = ['keyword_args']
    is_temp = 1
    type = dict_type
    reject_duplicates = True

    def calculate_constant_result(self):
        result = {}
        reject_duplicates = self.reject_duplicates
        for item in self.keyword_args:
            if item.is_dict_literal:
                items = ((key.constant_result, value.constant_result) for key, value in item.key_value_pairs)
            else:
                items = item.constant_result.iteritems()
            for key, value in items:
                if reject_duplicates and key in result:
                    raise ValueError('duplicate keyword argument found: %s' % key)
                result[key] = value
        self.constant_result = result

    def compile_time_value(self, denv):
        result = {}
        reject_duplicates = self.reject_duplicates
        for item in self.keyword_args:
            if item.is_dict_literal:
                items = [(key.compile_time_value(denv), value.compile_time_value(denv)) for key, value in item.key_value_pairs]
            else:
                items = item.compile_time_value(denv).iteritems()
            try:
                for key, value in items:
                    if reject_duplicates and key in result:
                        raise ValueError('duplicate keyword argument found: %s' % key)
                    result[key] = value
            except Exception as e:
                self.compile_time_value_error(e)
        return result

    def type_dependencies(self, env):
        return ()

    def infer_type(self, env):
        return dict_type

    def analyse_types(self, env):
        self.keyword_args = [arg.analyse_types(env).coerce_to_pyobject(env).as_none_safe_node('argument after ** must be a mapping, not NoneType') for arg in self.keyword_args]
        return self

    def may_be_none(self):
        return False
    gil_message = 'Constructing Python dict'

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        args = iter(self.keyword_args)
        item = next(args)
        item.generate_evaluation_code(code)
        if item.type is not dict_type:
            code.putln('if (likely(PyDict_CheckExact(%s))) {' % item.py_result())
        if item.is_dict_literal:
            item.make_owned_reference(code)
            code.putln('%s = %s;' % (self.result(), item.py_result()))
            item.generate_post_assignment_code(code)
        else:
            code.putln('%s = PyDict_Copy(%s); %s' % (self.result(), item.py_result(), code.error_goto_if_null(self.result(), item.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
        if item.type is not dict_type:
            code.putln('} else {')
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectCallOneArg', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_PyObject_CallOneArg((PyObject*)&PyDict_Type, %s); %s' % (self.result(), item.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            item.generate_disposal_code(code)
            code.putln('}')
        item.free_temps(code)
        helpers = set()
        for item in args:
            if item.is_dict_literal:
                for arg in item.key_value_pairs:
                    arg.generate_evaluation_code(code)
                    if self.reject_duplicates:
                        code.putln('if (unlikely(PyDict_Contains(%s, %s))) {' % (self.result(), arg.key.py_result()))
                        helpers.add('RaiseDoubleKeywords')
                        code.putln('__Pyx_RaiseDoubleKeywordsError("function", %s); %s' % (arg.key.py_result(), code.error_goto(self.pos)))
                        code.putln('}')
                    code.put_error_if_neg(arg.key.pos, 'PyDict_SetItem(%s, %s, %s)' % (self.result(), arg.key.py_result(), arg.value.py_result()))
                    arg.generate_disposal_code(code)
                    arg.free_temps(code)
            else:
                item.generate_evaluation_code(code)
                if self.reject_duplicates:
                    helpers.add('MergeKeywords')
                    code.put_error_if_neg(item.pos, '__Pyx_MergeKeywords(%s, %s)' % (self.result(), item.py_result()))
                else:
                    helpers.add('RaiseMappingExpected')
                    code.putln('if (unlikely(PyDict_Update(%s, %s) < 0)) {' % (self.result(), item.py_result()))
                    code.putln('if (PyErr_ExceptionMatches(PyExc_AttributeError)) __Pyx_RaiseMappingExpectedError(%s);' % item.py_result())
                    code.putln(code.error_goto(item.pos))
                    code.putln('}')
                item.generate_disposal_code(code)
                item.free_temps(code)
        for helper in sorted(helpers):
            code.globalstate.use_utility_code(UtilityCode.load_cached(helper, 'FunctionArguments.c'))

    def annotate(self, code):
        for item in self.keyword_args:
            item.annotate(code)