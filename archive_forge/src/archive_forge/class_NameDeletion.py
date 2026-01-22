from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class NameDeletion(NameAssignment):

    def __init__(self, lhs, entry):
        NameAssignment.__init__(self, lhs, lhs, entry)
        self.is_deletion = True

    def infer_type(self):
        inferred_type = self.rhs.infer_type(self.entry.scope)
        if not inferred_type.is_pyobject and inferred_type.can_coerce_to_pyobject(self.entry.scope):
            return PyrexTypes.py_object_type
        self.inferred_type = inferred_type
        return inferred_type