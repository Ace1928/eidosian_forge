from glob import glob
from os.path import basename, dirname
from os.path import join as pjoin
from warnings import simplefilter
import numpy as np
import pytest
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import load as top_load
from .. import parrec
from ..fileholders import FileHolder
from ..nifti1 import Nifti1Extension, Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..parrec import (
from ..testing import assert_arr_dict_equal, clear_and_catch_warnings, suppress_warnings
from ..volumeutils import array_from_file
from . import test_spatialimages as tsi
from .test_arrayproxy import check_mmap
class FakeHeader:
    """Minimal API of header for PARRECArrayProxy"""

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = np.dtype(dtype)

    def get_data_shape(self):
        return self._shape

    def get_data_dtype(self):
        return self._dtype

    def get_sorted_slice_indices(self):
        n_slices = np.prod(self._shape[2:])
        return np.arange(n_slices)

    def get_data_scaling(self, scaling):
        scale_shape = (1, 1) + self._shape[2:]
        return (np.ones(scale_shape), np.zeros(scale_shape))

    def get_rec_shape(self):
        n_slices = np.prod(self._shape[2:])
        return self._shape[:2] + (n_slices,)