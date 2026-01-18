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
def test_vol_matching():
    dw_siemens = didw.wrapper_from_data(DATA)
    assert dw_siemens.is_mosaic
    assert dw_siemens.is_csa
    assert dw_siemens.is_same_series(dw_siemens)
    dw_plain = didw.Wrapper(DATA)
    assert not dw_plain.is_mosaic
    assert not dw_plain.is_csa
    assert dw_plain.is_same_series(dw_plain)
    assert not dw_plain.is_same_series(dw_siemens)
    assert not dw_siemens.is_same_series(dw_plain)
    dw_empty = didw.Wrapper({})
    assert dw_empty.is_same_series(dw_empty)
    assert not dw_empty.is_same_series(dw_plain)
    assert not dw_plain.is_same_series(dw_empty)

    class C:
        series_signature = {}
    assert dw_empty.is_same_series(C())
    dw_philips = didw.wrapper_from_data(DATA_PHILIPS)
    assert dw_philips.is_multiframe
    assert dw_philips.is_same_series(dw_philips)
    dw_plain_philips = didw.Wrapper(DATA)
    assert not dw_plain_philips.is_multiframe
    assert dw_plain_philips.is_same_series(dw_plain_philips)
    assert not dw_plain_philips.is_same_series(dw_philips)
    assert not dw_philips.is_same_series(dw_plain_philips)
    dw_empty = didw.Wrapper({})
    assert dw_empty.is_same_series(dw_empty)
    assert not dw_empty.is_same_series(dw_plain_philips)
    assert not dw_plain_philips.is_same_series(dw_empty)