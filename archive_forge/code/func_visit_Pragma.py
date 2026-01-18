from . import c_ast
def visit_Pragma(self, n):
    ret = '#pragma'
    if n.string:
        ret += ' ' + n.string
    return ret