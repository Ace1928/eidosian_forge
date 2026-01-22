from . import c_ast
def _is_simple_node(self, n):
    """ Returns True for nodes that are "simple" - i.e. nodes that always
            have higher precedence than operators.
        """
    return isinstance(n, (c_ast.Constant, c_ast.ID, c_ast.ArrayRef, c_ast.StructRef, c_ast.FuncCall))