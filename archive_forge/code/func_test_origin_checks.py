import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def test_origin_checks(self):
    HC = self.header_class
    hdr = HC()
    hdr.data_shape = [1, 1, 1]
    hdr['origin'][0] = 101
    fhdr, message, raiser = self.log_chk(hdr, 20)
    assert fhdr == hdr
    assert message == 'very large origin values relative to dims; leaving as set, ignoring for affine'
    pytest.raises(*raiser)
    dxer = self.header_class.diagnose_binaryblock
    assert dxer(hdr.binaryblock) == 'very large origin values relative to dims'