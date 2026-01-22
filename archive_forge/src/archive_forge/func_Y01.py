import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
def Y01(theta, phi):
    return 0.5 * np.sqrt(3 / np.pi) * np.cos(phi)