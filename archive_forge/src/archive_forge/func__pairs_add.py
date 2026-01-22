import re
import warnings
from enum import Enum
from math import gcd
def _pairs_add(d, k, v):
    c = d.get(k)
    if c is None:
        d[k] = v
    else:
        c = c + v
        if c:
            d[k] = c
        else:
            del d[k]