from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class ArgminmaxPattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('numpy'), ctx=ast.Load(), annotation=None, type_comment=None), attr=Placeholder(0, constraint=lambda s: s in ('argmax', 'argmin')), ctx=ast.Load()), args=[AST_or(ast.BinOp(ast.Constant(Placeholder(2, constraint=lambda n: n > 0), None), ast.Mult(), Placeholder(1)), ast.BinOp(Placeholder(1), ast.Mult(), ast.Constant(Placeholder(2, constraint=lambda n: n > 0), None)))], keywords=[])

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('numpy'), ctx=ast.Load(), annotation=None, type_comment=None), attr=Placeholder(0), ctx=ast.Load()), args=[Placeholder(1)], keywords=[])