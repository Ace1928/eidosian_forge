import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_ornt2axcodes():
    labels = (('left', 'right'), ('back', 'front'), ('down', 'up'))
    assert ornt2axcodes([[0, 1], [1, 1], [2, 1]], labels) == ('right', 'front', 'up')
    assert ornt2axcodes([[0, -1], [1, -1], [2, -1]], labels) == ('left', 'back', 'down')
    assert ornt2axcodes([[2, -1], [1, -1], [0, -1]], labels) == ('down', 'back', 'left')
    assert ornt2axcodes([[1, 1], [2, -1], [0, 1]], labels) == ('front', 'down', 'right')
    assert ornt2axcodes([[0, 1], [1, 1], [2, 1]]) == ('R', 'A', 'S')
    assert ornt2axcodes([[0, 1], [np.nan, np.nan], [2, 1]]) == ('R', None, 'S')
    with pytest.raises(ValueError):
        ornt2axcodes([[0.1, 1]])
    with pytest.raises(ValueError):
        ornt2axcodes([[0, 0]])
    for axcodes, ornt in zip(ALL_AXCODES, ALL_ORNTS):
        assert ornt2axcodes(ornt) == axcodes