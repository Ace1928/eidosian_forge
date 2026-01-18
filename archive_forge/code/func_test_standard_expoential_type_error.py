import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_standard_expoential_type_error(self):
    assert_raises(TypeError, random.standard_exponential, dtype=np.int32)