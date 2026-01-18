from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.optpkg import optional_package
from .. import dicomreaders as didr
from .test_dicomwrappers import DATA, EXPECTED_AFFINE, EXPECTED_PARAMS, IO_DATA_PATH
def test_slices_to_series():
    dicom_files = (pjoin(IO_DATA_PATH, f'{i}.dcm') for i in range(2))
    wrappers = [didr.wrapper_from_file(f) for f in dicom_files]
    series = didr.slices_to_series(wrappers)
    assert len(series) == 1
    assert len(series[0]) == 2