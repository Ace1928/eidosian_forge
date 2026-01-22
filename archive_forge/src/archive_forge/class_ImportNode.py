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
class ImportNode(ExprNode):
    type = py_object_type
    module_names = None
    get_top_level_module = False
    is_temp = True
    subexprs = ['module_name', 'name_list', 'module_names']

    def analyse_types(self, env):
        if self.level is None:
            if env.global_scope().parent_module and (env.directives['py2_import'] or Future.absolute_import not in env.global_scope().context.future_directives):
                self.level = -1
            else:
                self.level = 0
        module_name = self.module_name.analyse_types(env)
        self.module_name = module_name.coerce_to_pyobject(env)
        assert self.module_name.is_string_literal
        if self.name_list:
            name_list = self.name_list.analyse_types(env)
            self.name_list = name_list.coerce_to_pyobject(env)
        elif '.' in self.module_name.value:
            self.module_names = TupleNode(self.module_name.pos, args=[IdentifierStringNode(self.module_name.pos, value=part, constant_result=part) for part in map(StringEncoding.EncodedString, self.module_name.value.split('.'))]).analyse_types(env)
        return self
    gil_message = 'Python import'

    def generate_result_code(self, code):
        assert self.module_name.is_string_literal
        module_name = self.module_name.value
        if self.level <= 0 and (not self.name_list) and (not self.get_top_level_module):
            if self.module_names:
                assert self.module_names.is_literal
            if self.level == 0:
                utility_code = UtilityCode.load_cached('ImportDottedModule', 'ImportExport.c')
                helper_func = '__Pyx_ImportDottedModule'
            else:
                utility_code = UtilityCode.load_cached('ImportDottedModuleRelFirst', 'ImportExport.c')
                helper_func = '__Pyx_ImportDottedModuleRelFirst'
            code.globalstate.use_utility_code(utility_code)
            import_code = '%s(%s, %s)' % (helper_func, self.module_name.py_result(), self.module_names.py_result() if self.module_names else 'NULL')
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('Import', 'ImportExport.c'))
            import_code = '__Pyx_Import(%s, %s, %d)' % (self.module_name.py_result(), self.name_list.py_result() if self.name_list else '0', self.level)
        if self.level <= 0 and module_name in utility_code_for_imports:
            helper_func, code_name, code_file = utility_code_for_imports[module_name]
            code.globalstate.use_utility_code(UtilityCode.load_cached(code_name, code_file))
            import_code = '%s(%s)' % (helper_func, import_code)
        code.putln('%s = %s; %s' % (self.result(), import_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def get_known_standard_library_import(self):
        return self.module_name.value