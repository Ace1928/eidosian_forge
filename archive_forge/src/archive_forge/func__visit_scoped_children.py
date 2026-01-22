from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def _visit_scoped_children(self, node, gil_state):
    was_nogil = self.nogil
    outer_attrs = node.outer_attrs
    if outer_attrs and len(self.env_stack) > 1:
        self.nogil = self.env_stack[-2].nogil
        self.visitchildren(node, outer_attrs)
    self.nogil = gil_state
    self.visitchildren(node, attrs=None, exclude=outer_attrs)
    self.nogil = was_nogil