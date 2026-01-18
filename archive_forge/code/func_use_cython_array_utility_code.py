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
def use_cython_array_utility_code(env):
    cython_scope = env.global_scope().context.cython_scope
    cython_scope.load_cythonscope()
    cython_scope.viewscope.lookup('array_cwrapper').used = True