import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
@dicom_test
def test_use_csa_sign():
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    iop = dw.image_orient_patient
    dw.image_orient_patient = np.c_[iop[:, 1], iop[:, 0]]
    dw2 = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert np.allclose(dw.slice_normal, dw2.slice_normal)