import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
def test_year_attribute(self):
    expected = np.array(['1999', '2004', '1817', '2100', '2013', '1631'], dtype='datetime64[Y]')
    assert_array_equal(self.data['attr_year'], expected)