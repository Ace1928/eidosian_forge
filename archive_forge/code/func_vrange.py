from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
@require_all_args
def vrange(self):
    """
        Yields v_steps+1 SymPy numbers ranging from
        v_min to v_max.
        """
    d = (self.v_max - self.v_min) / self.v_steps
    for i in range(self.v_steps + 1):
        a = self.v_min + d * Integer(i)
        yield a