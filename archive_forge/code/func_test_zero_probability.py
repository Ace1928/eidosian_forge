import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_zero_probability(self):
    random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])