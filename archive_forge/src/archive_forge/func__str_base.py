from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def _str_base(self):
    return ','.join([str(self.v), str(self.v_min), str(self.v_max), str(self.v_steps)])