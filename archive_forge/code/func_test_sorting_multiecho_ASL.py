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
def test_sorting_multiecho_ASL():
    asl_par = pjoin(DATA_PATH, 'ASL_3D_Multiecho.PAR')
    with open(asl_par) as fobj:
        asl_hdr = PARRECHeader.from_fileobj(fobj, strict_sort=True)
    np.random.shuffle(asl_hdr.image_defs)
    sorted_indices = asl_hdr.get_sorted_slice_indices()
    sorted_slices = asl_hdr.image_defs['slice number'][sorted_indices]
    sorted_echos = asl_hdr.image_defs['echo number'][sorted_indices]
    sorted_dynamics = asl_hdr.image_defs['dynamic scan number'][sorted_indices]
    sorted_labels = asl_hdr.image_defs['label type'][sorted_indices]
    ntotal = len(asl_hdr.image_defs)
    nslices = sorted_slices.max()
    nechos = sorted_echos.max()
    nlabels = sorted_labels.max()
    ndynamics = sorted_dynamics.max()
    assert nslices == 8
    assert nechos == 3
    assert nlabels == 2
    assert ndynamics == 2
    assert_array_equal(np.all(sorted_dynamics[:ntotal // ndynamics] == 1), True)
    assert_array_equal(np.all(sorted_dynamics[ntotal // ndynamics:ntotal] == 2), True)
    assert_array_equal(np.all(sorted_labels[:nslices * nechos] == 1), True)
    assert_array_equal(np.all(sorted_labels[nslices * nechos:2 * nslices * nechos] == 2), True)
    assert_array_equal(np.all(sorted_echos[:nslices] == 1), True)
    assert_array_equal(np.all(sorted_echos[nslices:2 * nslices] == 2), True)
    assert_array_equal(np.all(sorted_echos[2 * nslices:3 * nslices] == 3), True)
    assert_array_equal(sorted_slices[:nslices], np.arange(1, nslices + 1))
    vol_labels = asl_hdr.get_volume_labels()
    assert list(vol_labels.keys()) == ['echo number', 'label type', 'dynamic scan number']
    assert_array_equal(vol_labels['dynamic scan number'], [1] * 6 + [2] * 6)
    assert_array_equal(vol_labels['label type'], [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3)
    assert_array_equal(vol_labels['echo number'], [1, 2, 3] * 4)