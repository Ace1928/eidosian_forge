import os
import numpy as np
from numpy.testing import assert_array_equal
from .. import nifti2
from ..nifti1 import Nifti1Extension, Nifti1Extensions, Nifti1Header, Nifti1PairHeader
from ..nifti2 import Nifti2Header, Nifti2Image, Nifti2Pair, Nifti2PairHeader
from ..testing import data_path
from . import test_nifti1 as tn1
class TestNifti2Image(tn1.TestNifti1Image):
    image_class = Nifti2Image