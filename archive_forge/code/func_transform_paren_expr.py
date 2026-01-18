from sympy.external import import_module
import os
def transform_paren_expr(self, node):
    """Transformation function for Parenthesized expressions

            Returns the result from its children nodes

            """
    return self.transform(next(node.get_children()))