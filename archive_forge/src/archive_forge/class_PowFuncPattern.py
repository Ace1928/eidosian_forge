from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class PowFuncPattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('builtins'), ctx=ast.Load(), annotation=None, type_comment=None), attr='pow', ctx=ast.Load()), args=[Placeholder(0), Placeholder(1)], keywords=[])

    @staticmethod
    def sub():
        return ast.BinOp(Placeholder(0), ast.Pow(), Placeholder(1))