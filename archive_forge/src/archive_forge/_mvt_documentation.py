import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to

    Computes permuted and scaled lower Cholesky factor c for R which may be
    singular, also permuting and scaling integration limit vectors a and b.
    