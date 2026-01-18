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
def test_wrapper_from_data():
    for dw in (didw.wrapper_from_data(DATA), didw.wrapper_from_file(DATA_FILE)):
        assert dw.get('InstanceNumber') == 2
        assert dw.get('AcquisitionNumber') == 2
        with pytest.raises(KeyError):
            dw['not an item']
        assert dw.is_mosaic
        assert_array_almost_equal(np.dot(didr.DPCS_TO_TAL, dw.affine), EXPECTED_AFFINE)
    for dw in (didw.wrapper_from_data(DATA_PHILIPS), didw.wrapper_from_file(DATA_FILE_PHILIPS)):
        assert dw.get('InstanceNumber') == 1
        assert dw.get('AcquisitionNumber') == 3
        with pytest.raises(KeyError):
            dw['not an item']
        assert dw.is_multiframe
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert dw.is_mosaic
    fake_data = dict()
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.2'
    dw = didw.wrapper_from_data(fake_data)
    assert not dw.is_multiframe
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.1'
    with pytest.raises(didw.WrapperError):
        didw.wrapper_from_data(fake_data)
    fake_data['PerFrameFunctionalGroupsSequence'] = [None]
    with pytest.raises(didw.WrapperError):
        didw.wrapper_from_data(fake_data)
    fake_data['SharedFunctionalGroupsSequence'] = [None]
    dw = didw.wrapper_from_data(fake_data)
    assert dw.is_multiframe