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
def visit_cython_attribute(self, node):
    attribute = node.as_cython_attribute()
    if attribute:
        if attribute == u'__version__':
            from .. import __version__ as version
            node = ExprNodes.StringNode(node.pos, value=EncodedString(version))
        elif attribute == u'NULL':
            node = ExprNodes.NullNode(node.pos)
        elif attribute in (u'set', u'frozenset', u'staticmethod'):
            node = ExprNodes.NameNode(node.pos, name=EncodedString(attribute), entry=self.current_env().builtin_scope().lookup_here(attribute))
        elif PyrexTypes.parse_basic_type(attribute):
            pass
        elif self.context.cython_scope.lookup_qualified_name(attribute):
            pass
        else:
            error(node.pos, u"'%s' not a valid cython attribute or is being used incorrectly" % attribute)
    return node