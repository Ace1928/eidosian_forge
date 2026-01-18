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
def try_to_parse_directives(self, node):
    if isinstance(node, ExprNodes.CallNode):
        self.visitchild(node, 'function')
        optname = node.function.as_cython_attribute()
        if optname:
            directivetype = Options.directive_types.get(optname)
            if directivetype:
                args, kwds = node.explicit_args_kwds()
                directives = []
                key_value_pairs = []
                if kwds is not None and directivetype is not dict:
                    for keyvalue in kwds.key_value_pairs:
                        key, value = keyvalue
                        sub_optname = '%s.%s' % (optname, key.value)
                        if Options.directive_types.get(sub_optname):
                            directives.append(self.try_to_parse_directive(sub_optname, [value], None, keyvalue.pos))
                        else:
                            key_value_pairs.append(keyvalue)
                    if not key_value_pairs:
                        kwds = None
                    else:
                        kwds.key_value_pairs = key_value_pairs
                    if directives and (not kwds) and (not args):
                        return directives
                directives.append(self.try_to_parse_directive(optname, args, kwds, node.function.pos))
                return directives
    elif isinstance(node, (ExprNodes.AttributeNode, ExprNodes.NameNode)):
        self.visit(node)
        optname = node.as_cython_attribute()
        if optname:
            directivetype = Options.directive_types.get(optname)
            if directivetype is bool:
                arg = ExprNodes.BoolNode(node.pos, value=True)
                return [self.try_to_parse_directive(optname, [arg], None, node.pos)]
            elif directivetype is None or directivetype is Options.DEFER_ANALYSIS_OF_ARGUMENTS:
                return [(optname, None)]
            else:
                raise PostParseError(node.pos, "The '%s' directive should be used as a function call." % optname)
    return None