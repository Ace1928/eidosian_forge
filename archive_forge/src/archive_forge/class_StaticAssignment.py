from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class StaticAssignment(NameAssignment):
    """Initialised at declaration time, e.g. stack allocation."""

    def __init__(self, entry):
        if not entry.type.is_pyobject:
            may_be_none = False
        else:
            may_be_none = None
        lhs = TypedExprNode(entry.type, may_be_none=may_be_none, pos=entry.pos)
        super(StaticAssignment, self).__init__(lhs, lhs, entry)

    def infer_type(self):
        return self.entry.type

    def type_dependencies(self):
        return ()