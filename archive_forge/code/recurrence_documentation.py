from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify

        Multiplies two Operators and returns another
        RecurrenceOperator instance using the commutation rule
        Sn * a(n) = a(n + 1) * Sn
        