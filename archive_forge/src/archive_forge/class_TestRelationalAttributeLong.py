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
class TestRelationalAttributeLong:

    def setup_method(self):
        self.data, self.meta = loadarff(test10)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 1)
        relational = list(self.meta._attributes.values())[0]
        assert_equal(relational.name, 'attr_relational')
        assert_equal(relational.type_name, 'relational')
        assert_equal(len(relational.attributes), 1)
        assert_equal(relational.attributes[0].name, 'attr_number')
        assert_equal(relational.attributes[0].type_name, 'numeric')

    def test_data(self):
        dtype_instance = [('attr_number', np.float64)]
        expected = np.array([(n,) for n in range(30000)], dtype=dtype_instance)
        assert_array_equal(self.data['attr_relational'][0], expected)