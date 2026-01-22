from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
class CommaOperator(Basic):
    """ Represents the comma operator in C """

    def __new__(cls, *args):
        return Basic.__new__(cls, *[sympify(arg) for arg in args])