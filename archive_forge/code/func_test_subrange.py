from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def test_subrange():
    mult = [4, 4, 2, 1]
    lb = 1
    ub = 2
    subrange_exercise(mult, lb, ub)