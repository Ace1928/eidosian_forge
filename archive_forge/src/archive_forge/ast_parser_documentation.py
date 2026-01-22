from sympy.core.basic import Basic
from sympy.core.sympify import SympifyError
from ast import parse, NodeTransformer, Call, Name, Load, \

    Converts the string "s" to a SymPy expression, in local_dict.

    It converts all numbers to Integers before feeding it to Python and
    automatically creates Symbols.
    