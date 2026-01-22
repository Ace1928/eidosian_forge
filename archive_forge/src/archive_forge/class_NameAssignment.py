from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class NameAssignment(object):

    def __init__(self, lhs, rhs, entry, rhs_scope=None):
        if lhs.cf_state is None:
            lhs.cf_state = set()
        self.lhs = lhs
        self.rhs = rhs
        self.entry = entry
        self.pos = lhs.pos
        self.refs = set()
        self.is_arg = False
        self.is_deletion = False
        self.inferred_type = None
        self.rhs_scope = rhs_scope

    def __repr__(self):
        return '%s(entry=%r)' % (self.__class__.__name__, self.entry)

    def infer_type(self):
        self.inferred_type = self.rhs.infer_type(self.rhs_scope or self.entry.scope)
        return self.inferred_type

    def type_dependencies(self):
        return self.rhs.type_dependencies(self.rhs_scope or self.entry.scope)

    @property
    def type(self):
        if not self.entry.type.is_unspecified:
            return self.entry.type
        return self.inferred_type