from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
def _l_p_exact(L, m, n):
    """Calculate the p-value of Page's L exactly"""
    L, n, k = (int(L), int(m), int(n))
    _pagel_state.set_k(k)
    return _pagel_state.sf(L, n)