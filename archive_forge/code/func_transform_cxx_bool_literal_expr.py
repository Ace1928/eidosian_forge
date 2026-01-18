from sympy.external import import_module
import os
def transform_cxx_bool_literal_expr(self, node):
    """Transformation function for boolean literal

            Used to get the value of the given boolean literal.

            Returns
            =======

            value : bool
                value contains the boolean value of the variable

            """
    try:
        value = next(node.get_tokens()).spelling
    except (StopIteration, ValueError):
        value = node.literal
    return True if value == 'true' else False