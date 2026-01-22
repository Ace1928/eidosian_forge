import math
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type, check_nD
def _sigma_prefactor(bandwidth):
    b = bandwidth
    return 1.0 / np.pi * math.sqrt(math.log(2) / 2.0) * (2.0 ** b + 1) / (2.0 ** b - 1)