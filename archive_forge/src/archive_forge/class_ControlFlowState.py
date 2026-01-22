from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class ControlFlowState(list):
    cf_maybe_null = False
    cf_is_null = False
    is_single = False

    def __init__(self, state):
        if Uninitialized in state:
            state.discard(Uninitialized)
            self.cf_maybe_null = True
            if not state:
                self.cf_is_null = True
        elif Unknown in state:
            state.discard(Unknown)
            self.cf_maybe_null = True
        elif len(state) == 1:
            self.is_single = True
        super(ControlFlowState, self).__init__([i for i in state if i.rhs is not fake_rhs_expr])

    def one(self):
        return self[0]