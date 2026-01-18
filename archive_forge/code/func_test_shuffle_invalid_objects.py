import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_shuffle_invalid_objects(self):
    x = np.array(3)
    assert_raises(TypeError, random.shuffle, x)