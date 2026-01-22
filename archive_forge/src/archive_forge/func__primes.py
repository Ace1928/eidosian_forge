import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to
def _primes(n):
    return primes_from_2_to(math.ceil(n))