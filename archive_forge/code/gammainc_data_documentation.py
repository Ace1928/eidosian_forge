import os
from time import time
import numpy as np
from numpy import pi
from scipy.special._mptestutils import mpf2float
Compute gammaincc exactly like mpmath does but allow for more
    terms in hypercomb. See

    mpmath/functions/expintegrals.py#L187

    in the mpmath github repository.

    