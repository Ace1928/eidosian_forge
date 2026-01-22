from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
def _choose_method(ranks):
    """Choose method for computing p-value automatically"""
    m, n = ranks.shape
    if n > 8 or (m > 12 and n > 3) or m > 20:
        method = 'asymptotic'
    else:
        method = 'exact'
    return method