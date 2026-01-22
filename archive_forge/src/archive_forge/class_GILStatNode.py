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
class GILStatNode(NogilTryFinallyStatNode):
    child_attrs = ['condition'] + NogilTryFinallyStatNode.child_attrs
    state_temp = None
    scope_gil_state_known = True

    def __init__(self, pos, state, body, condition=None):
        self.state = state
        self.condition = condition
        self.create_state_temp_if_needed(pos, state, body)
        TryFinallyStatNode.__init__(self, pos, body=body, finally_clause=GILExitNode(pos, state=state, state_temp=self.state_temp))

    def create_state_temp_if_needed(self, pos, state, body):
        from .ParseTreeTransforms import YieldNodeCollector
        collector = YieldNodeCollector()
        collector.visitchildren(body)
        if not collector.yields:
            return
        if state == 'gil':
            temp_type = PyrexTypes.c_gilstate_type
        else:
            temp_type = PyrexTypes.c_threadstate_ptr_type
        from . import ExprNodes
        self.state_temp = ExprNodes.TempNode(pos, temp_type)

    def analyse_declarations(self, env):
        env._in_with_gil_block = self.state == 'gil'
        if self.state == 'gil':
            env.has_with_gil_block = True
        if self.condition is not None:
            self.condition.analyse_declarations(env)
        return super(GILStatNode, self).analyse_declarations(env)

    def analyse_expressions(self, env):
        env.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
        if self.condition is not None:
            self.condition = self.condition.analyse_expressions(env)
        was_nogil = env.nogil
        env.nogil = self.state == 'nogil'
        node = TryFinallyStatNode.analyse_expressions(self, env)
        env.nogil = was_nogil
        return node

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.begin_block()
        if self.state_temp:
            self.state_temp.allocate(code)
            variable = self.state_temp.result()
        else:
            variable = None
        old_gil_config = code.funcstate.gil_owned
        if self.state == 'gil':
            code.put_ensure_gil(variable=variable)
            code.funcstate.gil_owned = True
        else:
            code.put_release_gil(variable=variable, unknown_gil_state=not self.scope_gil_state_known)
            code.funcstate.gil_owned = False
        TryFinallyStatNode.generate_execution_code(self, code)
        if self.state_temp:
            self.state_temp.release(code)
        code.funcstate.gil_owned = old_gil_config
        code.end_block()