from sympy.tensor import Indexed
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.integrals.integrals import Integral
@staticmethod
def _indexed_process_limits(limits):
    repl = {}
    newlimits = []
    for i in limits:
        if isinstance(i, (tuple, list, Tuple)):
            v = i[0]
            vrest = i[1:]
        else:
            v = i
            vrest = ()
        if isinstance(v, Indexed):
            if v not in repl:
                r = Dummy(str(v))
                repl[v] = r
            newlimits.append((r,) + vrest)
        else:
            newlimits.append(i)
    return (repl, newlimits)