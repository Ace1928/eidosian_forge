from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
@classmethod
def _construct_body(cls, itr):
    if isinstance(itr, CodeBlock):
        return itr
    else:
        return CodeBlock(*itr)