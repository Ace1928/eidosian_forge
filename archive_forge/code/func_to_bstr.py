import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def to_bstr(x):
    sign, man, exp, bc = x
    return ['', '-'][sign] + numeral(man, size=bitcount(man), base=2) + 'e%i' % exp