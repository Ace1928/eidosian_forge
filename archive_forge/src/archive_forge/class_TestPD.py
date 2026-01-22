import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
class TestPD(TestEPIFrame):
    example_params = TestT1.example_params.copy()
    new_params = dict(fname=os.path.join(MINC2_PATH, 'mincex_pd.mnc'), min=0, max=102.5024482, mean=23.82625718)
    example_params.update(new_params)