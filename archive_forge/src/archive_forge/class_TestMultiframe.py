import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from ..ecat import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestMultiframe(TestNegatives):
    example_params = dict(fname=os.path.join(ECAT_TEST_PATH, 'ECAT7_testcase_multiframe.v'), shape=(256, 256, 207, 3), type=np.int16, min=0.0, max=29170.67905, mean=121.454)