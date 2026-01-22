import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to
def _Phinv(p):
    return special.ndtri(p)