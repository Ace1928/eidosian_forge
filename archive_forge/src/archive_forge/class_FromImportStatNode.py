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
class FromImportStatNode(StatNode):
    child_attrs = ['module']
    import_star = 0

    def analyse_declarations(self, env):
        for name, target in self.items:
            if name == '*':
                if not env.is_module_scope:
                    error(self.pos, 'import * only allowed at module level')
                    return
                env.has_import_star = 1
                self.import_star = 1
            else:
                target.analyse_target_declaration(env)
                if target.entry:
                    if target.get_known_standard_library_import() is None:
                        target.entry.known_standard_library_import = EncodedString('%s.%s' % (self.module.module_name.value, name))
                else:
                    target.entry.known_standard_library_import = ''

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.module = self.module.analyse_expressions(env)
        self.item = ExprNodes.RawCNameExprNode(self.pos, py_object_type)
        self.interned_items = []
        for name, target in self.items:
            if name == '*':
                for _, entry in env.entries.items():
                    if not entry.is_type and entry.type.is_extension_type:
                        env.use_utility_code(UtilityCode.load_cached('ExtTypeTest', 'ObjectHandling.c'))
                        break
            else:
                entry = env.lookup(target.name)
                if entry.is_type and entry.type.name == name and hasattr(entry.type, 'module_name'):
                    if entry.type.module_name == self.module.module_name.value:
                        continue
                    try:
                        module = env.find_module(self.module.module_name.value, pos=self.pos, relative_level=self.module.level)
                        if entry.type.module_name == module.qualified_name:
                            continue
                    except AttributeError:
                        pass
                target = target.analyse_target_expression(env, None)
                if target.type is py_object_type:
                    coerced_item = None
                else:
                    coerced_item = self.item.coerce_to(target.type, env)
                self.interned_items.append((name, target, coerced_item))
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.module.generate_evaluation_code(code)
        if self.import_star:
            code.putln('if (%s(%s) < 0) %s;' % (Naming.import_star, self.module.py_result(), code.error_goto(self.pos)))
        item_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        self.item.set_cname(item_temp)
        if self.interned_items:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ImportFrom', 'ImportExport.c'))
        for name, target, coerced_item in self.interned_items:
            code.putln('%s = __Pyx_ImportFrom(%s, %s); %s' % (item_temp, self.module.py_result(), code.intern_identifier(name), code.error_goto_if_null(item_temp, self.pos)))
            code.put_gotref(item_temp, py_object_type)
            if coerced_item is None:
                target.generate_assignment_code(self.item, code)
            else:
                coerced_item.allocate_temp_result(code)
                coerced_item.generate_result_code(code)
                target.generate_assignment_code(coerced_item, code)
            code.put_decref_clear(item_temp, py_object_type)
        code.funcstate.release_temp(item_temp)
        self.module.generate_disposal_code(code)
        self.module.free_temps(code)