from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
@classmethod
def _construct_declarations(cls, args):
    return Tuple(*[Declaration(arg) for arg in args])