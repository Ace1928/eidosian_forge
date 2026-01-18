from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
from sympy.printing.mathematica import mathematica_code as mcode
def test_userfuncs():
    some_function = symbols('some_function', cls=Function)
    my_user_functions = {'some_function': 'SomeFunction'}
    assert mcode(some_function(z), user_functions=my_user_functions) == 'SomeFunction[z]'
    assert mcode(some_function(z), user_functions=my_user_functions) == 'SomeFunction[z]'
    my_user_functions = {'some_function': [(lambda x: True, 'SomeOtherFunction')]}
    assert mcode(some_function(z), user_functions=my_user_functions) == 'SomeOtherFunction[z]'