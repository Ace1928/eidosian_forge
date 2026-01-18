from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def visit_assignment(self, lhs, rhs):
    if isinstance(rhs, ExprNodes.IntNode) and isinstance(lhs, ExprNodes.NameNode) and Utils.long_literal(rhs.value):
        entry = lhs.entry or self.env.lookup(lhs.name)
        if entry:
            entry.might_overflow = True