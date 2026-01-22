import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def _assert_arrays_equal(self, expected, actual):
    self.assertEqual(expected.shape, actual.shape)
    self.assertTrue((actual == expected).all())