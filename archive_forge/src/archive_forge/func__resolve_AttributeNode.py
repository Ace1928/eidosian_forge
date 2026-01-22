from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def _resolve_AttributeNode(env, node):
    path = []
    while isinstance(node, AttributeNode):
        path.insert(0, node.attribute)
        node = node.obj
    if isinstance(node, NameNode):
        path.insert(0, node.name)
    else:
        raise CompileError(node.pos, EXPR_ERR)
    modnames = path[:-1]
    assert modnames
    scope = env
    for modname in modnames:
        mod = scope.lookup(modname)
        if not mod or not mod.as_module:
            raise CompileError(node.pos, 'undeclared name not builtin: %s' % modname)
        scope = mod.as_module
    entry = scope.lookup(path[-1])
    if not entry:
        raise CompileError(node.pos, "No such attribute '%s'" % path[-1])
    return entry