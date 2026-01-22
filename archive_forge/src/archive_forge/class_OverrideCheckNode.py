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
class OverrideCheckNode(StatNode):
    child_attrs = ['body']
    body = None

    def analyse_expressions(self, env):
        self.args = env.arg_entries
        if self.py_func.is_module_scope:
            first_arg = 0
        else:
            first_arg = 1
        from . import ExprNodes
        self.func_node = ExprNodes.RawCNameExprNode(self.pos, py_object_type)
        call_node = ExprNodes.SimpleCallNode(self.pos, function=self.func_node, args=[ExprNodes.NameNode(self.pos, name=arg.name) for arg in self.args[first_arg:]])
        if env.return_type.is_void or env.return_type.is_returncode:
            self.body = StatListNode(self.pos, stats=[ExprStatNode(self.pos, expr=call_node), ReturnStatNode(self.pos, value=None)])
        else:
            self.body = ReturnStatNode(self.pos, value=call_node)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        method_entry = self.py_func.fused_py_func.entry if self.py_func.fused_py_func else self.py_func.entry
        interned_attr_cname = code.intern_identifier(method_entry.name)
        if self.py_func.is_module_scope:
            self_arg = '((PyObject *)%s)' % Naming.module_cname
        else:
            self_arg = '((PyObject *)%s)' % self.args[0].cname
        code.putln('/* Check if called by wrapper */')
        code.putln('if (unlikely(%s)) ;' % Naming.skip_dispatch_cname)
        code.putln('/* Check if overridden in Python */')
        if self.py_func.is_module_scope:
            code.putln('else {')
        else:
            code.putln('else if (unlikely((Py_TYPE(%s)->tp_dictoffset != 0) || __Pyx_PyType_HasFeature(Py_TYPE(%s), (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)))) {' % (self_arg, self_arg))
        code.putln('#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS')
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyDictVersioning', 'ObjectHandling.c'))
        code.putln('static PY_UINT64_T %s = __PYX_DICT_VERSION_INIT, %s = __PYX_DICT_VERSION_INIT;' % (Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln('if (unlikely(!__Pyx_object_dict_version_matches(%s, %s, %s))) {' % (self_arg, Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln('PY_UINT64_T %s = __Pyx_get_tp_dict_version(%s);' % (Naming.type_dict_guard_temp, self_arg))
        code.putln('#endif')
        func_node_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        self.func_node.set_cname(func_node_temp)
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectGetAttrStr', 'ObjectHandling.c'))
        code.putln('%s = __Pyx_PyObject_GetAttrStr(%s, %s); %s' % (func_node_temp, self_arg, interned_attr_cname, code.error_goto_if_null(func_node_temp, self.pos)))
        code.put_gotref(func_node_temp, py_object_type)
        code.putln('if (!__Pyx_IsSameCFunction(%s, (void*) %s)) {' % (func_node_temp, method_entry.func_cname))
        self.body.generate_execution_code(code)
        code.putln('}')
        code.putln('#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS')
        code.putln('%s = __Pyx_get_tp_dict_version(%s);' % (Naming.tp_dict_version_temp, self_arg))
        code.putln('%s = __Pyx_get_object_dict_version(%s);' % (Naming.obj_dict_version_temp, self_arg))
        code.putln('if (unlikely(%s != %s)) {' % (Naming.type_dict_guard_temp, Naming.tp_dict_version_temp))
        code.putln('%s = %s = __PYX_DICT_VERSION_INIT;' % (Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln('}')
        code.putln('#endif')
        code.put_decref_clear(func_node_temp, PyrexTypes.py_object_type)
        code.funcstate.release_temp(func_node_temp)
        code.putln('#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS')
        code.putln('}')
        code.putln('#endif')
        code.putln('}')