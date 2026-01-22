from gast.astn import AstToGAst, GAstToAst
import gast
import ast
import sys
def adjust_slice(s):
    if isinstance(s, ast.Slice):
        return s
    else:
        return ast.Index(s)