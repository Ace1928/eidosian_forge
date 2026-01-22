from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
class goto(Token):
    """ Represents goto in C """
    __slots__ = _fields = ('label',)
    _construct_label = Label