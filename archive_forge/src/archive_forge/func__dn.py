import math
from sympy.utilities.misc import as_int
def _dn(n, prec):
    n += 1
    return int(math.log(n + prec) / math.log(16) + prec + 3)