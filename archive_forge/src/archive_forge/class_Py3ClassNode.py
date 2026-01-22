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
class Py3ClassNode(ExprNode):
    subexprs = []
    type = py_object_type
    force_type = False
    is_temp = True

    def infer_type(self, env):
        return py_object_type

    def analyse_types(self, env):
        return self

    def may_be_none(self):
        return True
    gil_message = 'Constructing Python class'

    def analyse_annotations(self, env):
        from .AutoDocTransforms import AnnotationWriter
        position = self.class_def_node.pos
        dict_items = [DictItemNode(entry.pos, key=IdentifierStringNode(entry.pos, value=entry.name), value=entry.annotation.string) for entry in env.entries.values() if entry.annotation]
        if dict_items:
            annotations_dict = DictNode(position, key_value_pairs=dict_items)
            lhs = NameNode(position, name=StringEncoding.EncodedString(u'__annotations__'))
            lhs.entry = env.lookup_here(lhs.name) or env.declare_var(lhs.name, dict_type, position)
            node = SingleAssignmentNode(position, lhs=lhs, rhs=annotations_dict)
            node.analyse_declarations(env)
            self.class_def_node.body.stats.insert(0, node)

    def generate_result_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached('Py3ClassCreate', 'ObjectHandling.c'))
        cname = code.intern_identifier(self.name)
        class_def_node = self.class_def_node
        mkw = class_def_node.mkw.py_result() if class_def_node.mkw else 'NULL'
        if class_def_node.metaclass:
            metaclass = class_def_node.metaclass.py_result()
        elif self.force_type:
            metaclass = '((PyObject*)&PyType_Type)'
        else:
            metaclass = '((PyObject*)&__Pyx_DefaultClassType)'
        code.putln('%s = __Pyx_Py3ClassCreate(%s, %s, %s, %s, %s, %d, %d); %s' % (self.result(), metaclass, cname, class_def_node.bases.py_result(), class_def_node.dict.py_result(), mkw, self.calculate_metaclass, self.allow_py2_metaclass, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)