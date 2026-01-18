import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_vonmises_small(self):
    random.seed(self.seed)
    r = random.vonmises(mu=0.0, kappa=1.1e-08, size=10 ** 6)
    assert_(np.isfinite(r).all())