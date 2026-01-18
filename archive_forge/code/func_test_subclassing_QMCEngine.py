import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_subclassing_QMCEngine():
    engine = RandomEngine(2, seed=175180605424926556207367152557812293274)
    sample_1 = engine.random(n=5)
    sample_2 = engine.random(n=7)
    assert engine.num_generated == 12
    engine.reset()
    assert engine.num_generated == 0
    sample_1_test = engine.random(n=5)
    assert_equal(sample_1, sample_1_test)
    engine.reset()
    engine.fast_forward(n=5)
    sample_2_test = engine.random(n=7)
    assert_equal(sample_2, sample_2_test)
    assert engine.num_generated == 12