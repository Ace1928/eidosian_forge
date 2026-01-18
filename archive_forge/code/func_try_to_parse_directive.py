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
def try_to_parse_directive(self, optname, args, kwds, pos):
    if optname == 'np_pythran' and (not self.context.cpp):
        raise PostParseError(pos, 'The %s directive can only be used in C++ mode.' % optname)
    elif optname == 'exceptval':
        arg_error = len(args) > 1
        check = True
        if kwds and kwds.key_value_pairs:
            kw = kwds.key_value_pairs[0]
            if len(kwds.key_value_pairs) == 1 and kw.key.is_string_literal and (kw.key.value == 'check') and isinstance(kw.value, ExprNodes.BoolNode):
                check = kw.value.value
            else:
                arg_error = True
        if arg_error:
            raise PostParseError(pos, 'The exceptval directive takes 0 or 1 positional arguments and the boolean keyword "check"')
        return ('exceptval', (args[0] if args else None, check))
    directivetype = Options.directive_types.get(optname)
    if len(args) == 1 and isinstance(args[0], ExprNodes.NoneNode):
        return (optname, Options.get_directive_defaults()[optname])
    elif directivetype is bool:
        if kwds is not None or len(args) != 1 or (not isinstance(args[0], ExprNodes.BoolNode)):
            raise PostParseError(pos, 'The %s directive takes one compile-time boolean argument' % optname)
        return (optname, args[0].value)
    elif directivetype is int:
        if kwds is not None or len(args) != 1 or (not isinstance(args[0], ExprNodes.IntNode)):
            raise PostParseError(pos, 'The %s directive takes one compile-time integer argument' % optname)
        return (optname, int(args[0].value))
    elif directivetype is str:
        if kwds is not None or len(args) != 1 or (not isinstance(args[0], (ExprNodes.StringNode, ExprNodes.UnicodeNode))):
            raise PostParseError(pos, 'The %s directive takes one compile-time string argument' % optname)
        return (optname, str(args[0].value))
    elif directivetype is type:
        if kwds is not None or len(args) != 1:
            raise PostParseError(pos, 'The %s directive takes one type argument' % optname)
        return (optname, args[0])
    elif directivetype is dict:
        if len(args) != 0:
            raise PostParseError(pos, 'The %s directive takes no prepositional arguments' % optname)
        return (optname, kwds.as_python_dict())
    elif directivetype is list:
        if kwds and len(kwds.key_value_pairs) != 0:
            raise PostParseError(pos, 'The %s directive takes no keyword arguments' % optname)
        return (optname, [str(arg.value) for arg in args])
    elif callable(directivetype):
        if kwds is not None or len(args) != 1 or (not isinstance(args[0], (ExprNodes.StringNode, ExprNodes.UnicodeNode))):
            raise PostParseError(pos, 'The %s directive takes one compile-time string argument' % optname)
        return (optname, directivetype(optname, str(args[0].value)))
    elif directivetype is Options.DEFER_ANALYSIS_OF_ARGUMENTS:
        return (optname, (args, kwds.as_python_dict() if kwds else {}))
    else:
        assert False