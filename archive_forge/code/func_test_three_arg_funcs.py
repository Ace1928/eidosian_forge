import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_three_arg_funcs(self):
    funcs = [random.noncentral_f, random.triangular, random.hypergeometric]
    for func in funcs:
        out = func(self.argOne, self.argTwo, self.argThree)
        assert_equal(out.shape, self.tgtShape)
        out = func(self.argOne[0], self.argTwo, self.argThree)
        assert_equal(out.shape, self.tgtShape)
        out = func(self.argOne, self.argTwo[0], self.argThree)
        assert_equal(out.shape, self.tgtShape)