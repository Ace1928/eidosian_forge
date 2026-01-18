from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
@require_all_args
def vrange2(self):
    """
        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
    d = (self.v_max - self.v_min) / self.v_steps
    a = self.v_min + d * S.Zero
    for i in range(self.v_steps):
        b = self.v_min + d * Integer(i + 1)
        yield (a, b)
        a = b