import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_vonmises_nan(self):
    random.seed(self.seed)
    r = random.vonmises(mu=0.0, kappa=np.nan)
    assert_(np.isnan(r))