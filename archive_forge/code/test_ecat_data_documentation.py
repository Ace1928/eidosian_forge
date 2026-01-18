import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from ..ecat import load
from .nibabel_data import get_nibabel_data, needs_nibabel_data
Test we can correctly import example ECAT files
