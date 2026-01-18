from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer

        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        