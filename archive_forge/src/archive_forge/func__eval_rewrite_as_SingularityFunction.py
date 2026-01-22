from sympy.core import S, diff
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent
def _eval_rewrite_as_SingularityFunction(self, args, H0=S.Half, **kwargs):
    """
        Returns the Heaviside expression written in the form of Singularity
        Functions.

        """
    from sympy.solvers import solve
    from sympy.functions.special.singularity_functions import SingularityFunction
    if self == Heaviside(0):
        return SingularityFunction(0, 0, 0)
    free = self.free_symbols
    if len(free) == 1:
        x = free.pop()
        return SingularityFunction(x, solve(args, x)[0], 0)
    else:
        raise TypeError(filldedent('\n                rewrite(SingularityFunction) does not\n                support arguments with more that one variable.'))