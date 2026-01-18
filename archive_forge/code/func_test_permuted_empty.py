import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permuted_empty(self):
    y = random.permuted([])
    assert_array_equal(y, [])