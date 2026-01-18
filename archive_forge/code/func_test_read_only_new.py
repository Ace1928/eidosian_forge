import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_read_only_new(self):
    self.result.plate_of_shrimp = 'lattice of coincidence'
    assert self.result.plate_of_shrimp == 'lattice of coincidence'