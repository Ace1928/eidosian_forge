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
def test_vol_calculations():
    for par, fobj in gen_par_fobj():
        gen_info, slice_info = parse_PAR_header(fobj)
        slice_nos = slice_info['slice number']
        max_slice = gen_info['max_slices']
        assert set(slice_nos) == set(range(1, max_slice + 1))
        assert_array_equal(vol_is_full(slice_nos, max_slice), True)
        if par.endswith('NA.PAR'):
            continue
        with suppress_warnings():
            hdr = PARRECHeader(gen_info, slice_info, True)
        shape = hdr.get_data_shape()
        d4 = 1 if len(shape) == 3 else shape[3]
        assert max(vol_numbers(slice_nos)) == d4 - 1