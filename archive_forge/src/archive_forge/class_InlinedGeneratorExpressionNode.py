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
class InlinedGeneratorExpressionNode(ExprNode):
    subexprs = ['gen']
    orig_func = None
    target = None
    is_temp = True
    type = py_object_type

    def __init__(self, pos, gen, comprehension_type=None, **kwargs):
        gbody = gen.def_node.gbody
        gbody.is_inlined = True
        if comprehension_type is not None:
            assert comprehension_type in (list_type, set_type, dict_type), comprehension_type
            gbody.inlined_comprehension_type = comprehension_type
            kwargs.update(target=RawCNameExprNode(pos, comprehension_type, Naming.retval_cname), type=comprehension_type)
        super(InlinedGeneratorExpressionNode, self).__init__(pos, gen=gen, **kwargs)

    def may_be_none(self):
        return self.orig_func not in ('any', 'all', 'sorted')

    def infer_type(self, env):
        return self.type

    def analyse_types(self, env):
        self.gen = self.gen.analyse_expressions(env)
        return self

    def generate_result_code(self, code):
        code.putln('%s = __Pyx_Generator_Next(%s); %s' % (self.result(), self.gen.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)