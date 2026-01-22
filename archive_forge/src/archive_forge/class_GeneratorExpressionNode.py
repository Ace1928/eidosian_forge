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
class GeneratorExpressionNode(LambdaNode):
    name = StringEncoding.EncodedString('genexpr')
    binding = False
    child_attrs = LambdaNode.child_attrs + ['call_parameters']
    subexprs = LambdaNode.subexprs + ['call_parameters']

    def __init__(self, pos, *args, **kwds):
        super(GeneratorExpressionNode, self).__init__(pos, *args, **kwds)
        self.call_parameters = []

    def analyse_declarations(self, env):
        if hasattr(self, 'genexpr_name'):
            return
        self.genexpr_name = env.next_id('genexpr')
        super(GeneratorExpressionNode, self).analyse_declarations(env)
        self.def_node.pymethdef_required = False
        self.def_node.py_wrapper_required = False
        self.def_node.is_cyfunction = False
        self.def_node.entry.signature = TypeSlots.pyfunction_noargs
        if isinstance(self.loop, Nodes._ForInStatNode):
            assert isinstance(self.loop.iterator, ScopedExprNode)
            self.loop.iterator.init_scope(None, env)
        else:
            assert isinstance(self.loop, Nodes.ForFromStatNode)

    def generate_result_code(self, code):
        args_to_call = [self.closure_result_code()] + [cp.result() for cp in self.call_parameters]
        args_to_call = ', '.join(args_to_call)
        code.putln('%s = %s(%s); %s' % (self.result(), self.def_node.entry.pyfunc_cname, args_to_call, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)