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
class TestSobol(QMCEngineTests):
    qmce = qmc.Sobol
    can_scramble = True
    unscramble_nd = np.array([[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.375, 0.375], [0.875, 0.875], [0.625, 0.125], [0.125, 0.625]])
    scramble_nd = np.array([[0.25331921, 0.41371179], [0.8654213, 0.9821167], [0.70097554, 0.03664616], [0.18027647, 0.60895735], [0.10521339, 0.21897069], [0.53019685, 0.66619033], [0.91122276, 0.34580743], [0.45337471, 0.78912079]])

    def test_warning(self):
        with pytest.warns(UserWarning, match="The balance properties of Sobol' points"):
            engine = qmc.Sobol(1)
            engine.random(10)

    def test_random_base2(self):
        engine = qmc.Sobol(2, scramble=False)
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[:4], sample)
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[4:8], sample)
        with pytest.raises(ValueError, match="The balance properties of Sobol' points"):
            engine.random_base2(2)

    def test_raise(self):
        with pytest.raises(ValueError, match='Maximum supported dimensionality'):
            qmc.Sobol(qmc.Sobol.MAXDIM + 1)
        with pytest.raises(ValueError, match="Maximum supported 'bits' is 64"):
            qmc.Sobol(1, bits=65)

    def test_high_dim(self):
        engine = qmc.Sobol(1111, scramble=False)
        count1 = Counter(engine.random().flatten().tolist())
        count2 = Counter(engine.random().flatten().tolist())
        assert_equal(count1, Counter({0.0: 1111}))
        assert_equal(count2, Counter({0.5: 1111}))

    @pytest.mark.parametrize('bits', [2, 3])
    def test_bits(self, bits):
        engine = qmc.Sobol(2, scramble=False, bits=bits)
        ns = 2 ** bits
        sample = engine.random(ns)
        assert_array_equal(self.unscramble_nd[:ns], sample)
        with pytest.raises(ValueError, match='increasing `bits`'):
            engine.random()

    def test_64bits(self):
        engine = qmc.Sobol(2, scramble=False, bits=64)
        sample = engine.random(8)
        assert_array_equal(self.unscramble_nd, sample)