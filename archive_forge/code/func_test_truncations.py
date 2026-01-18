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
def test_truncations():
    par = pjoin(DATA_PATH, 'T2_.PAR')
    with open(par) as fobj:
        gen_info, slice_info = parse_PAR_header(fobj)
    hdr = PARRECHeader(gen_info, slice_info)
    assert hdr.get_data_shape() == (80, 80, 10, 2)
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info[:-1])
    with clear_and_catch_warnings(modules=[parrec], record=True) as wlist:
        hdr = PARRECHeader(gen_info, slice_info[:-1], permit_truncated=True)
        assert len(wlist) == 1
    assert hdr.get_data_shape() == (80, 80, 10)
    gen_info['max_slices'] = 11
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info)
    gen_info['max_slices'] = 10
    hdr = PARRECHeader(gen_info, slice_info)
    gen_info['max_echoes'] = 2
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info)
    gen_info['max_echoes'] = 1
    hdr = PARRECHeader(gen_info, slice_info)
    gen_info['max_dynamics'] = 3
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info)
    gen_info['max_dynamics'] = 2
    hdr = PARRECHeader(gen_info, slice_info)
    gen_info['max_diffusion_values'] = 2
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info)
    gen_info['max_diffusion_values'] = 1
    hdr = PARRECHeader(gen_info, slice_info)
    gen_info['max_gradient_orient'] = 2
    with pytest.raises(PARRECError):
        PARRECHeader(gen_info, slice_info)
    gen_info['max_gradient_orient'] = 1
    hdr = PARRECHeader(gen_info, slice_info)