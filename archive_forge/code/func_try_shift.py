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
def try_shift(f, t, shifter, diff, counter):
    """ Try to apply ``shifter`` in order to bring some element in ``f``
            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and
            determines the effect of ``shifter``. Counter is a list of elements
            blocking the shift.

            Return an operator if change was possible, else None.
        """
    for idx, (a, b) in enumerate(zip(f, t)):
        if (a - b).is_integer and (b - a) / diff > 0 and all((a != x for x in counter)):
            sh = shifter(idx)
            f[idx] += diff
            return sh