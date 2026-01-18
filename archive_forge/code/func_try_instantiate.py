from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def try_instantiate(self, func):
    """
        Try to instantiate the current formula to (almost) match func.
        This uses the _matcher passed on init.
        """
    if func.signature != self.func.signature:
        return None
    res = self._matcher(func)
    if res is not None:
        subs, newfunc = res
        return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq, self.z, [], self.B.subs(subs), self.C.subs(subs), self.M.subs(subs), None)