from . import c_ast
def visit_Goto(self, n):
    return 'goto ' + n.name + ';'