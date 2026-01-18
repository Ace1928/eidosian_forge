from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
def test_unpack_var(self):
    for var in self.vars:
        with pytest.raises(self.module.AFNIHeaderError):
            self.module._unpack_var(var)