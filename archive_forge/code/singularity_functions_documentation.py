from sympy.core import S, oo, diff
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside

        Rewrites a Singularity Function expression using Heavisides and DiracDeltas.

        