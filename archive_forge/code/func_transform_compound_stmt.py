from sympy.external import import_module
import os
def transform_compound_stmt(self, node):
    """Transformation function for compond statemets

            Returns
            =======

            expr : list
                list of Nodes for the expressions present in the statement

            None : NoneType
                if the compound statement is empty

            """
    try:
        expr = []
        children = node.get_children()
        for child in children:
            expr.append(self.transform(child))
    except StopIteration:
        return None
    return expr