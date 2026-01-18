from sympy.external import import_module
import os
def transform_floating_literal(self, node):
    """Transformation function for floating literal

            Used to get the value and type of the given floating literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of float
                value contains the value stored in the variable

            Notes
            =====

            Only Base Float type supported for now

            """
    try:
        value = next(node.get_tokens()).spelling
    except (StopIteration, ValueError):
        value = node.literal
    return float(value)