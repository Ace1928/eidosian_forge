from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class TuplePattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='tuple', ctx=ast.Load()), args=[ast.List(Placeholder(0), ast.Load())], keywords=[])

    @staticmethod
    def sub():
        return ast.Tuple(Placeholder(0), ast.Load())