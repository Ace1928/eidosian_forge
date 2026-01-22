import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
def _decimal_sqrt_of_frac(n: int, m: int) -> Decimal:
    """Square root of n/m as a Decimal, correctly rounded."""
    if n <= 0:
        if not n:
            return Decimal('0.0')
        n, m = (-n, -m)
    root = (Decimal(n) / Decimal(m)).sqrt()
    nr, dr = root.as_integer_ratio()
    plus = root.next_plus()
    np, dp = plus.as_integer_ratio()
    if 4 * n * (dr * dp) ** 2 > m * (dr * np + dp * nr) ** 2:
        return plus
    minus = root.next_minus()
    nm, dm = minus.as_integer_ratio()
    if 4 * n * (dr * dm) ** 2 < m * (dr * nm + dm * nr) ** 2:
        return minus
    return root