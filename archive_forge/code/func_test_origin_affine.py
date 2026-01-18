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
def test_origin_affine():
    hdr = Spm99AnalyzeHeader()
    aff = hdr.get_origin_affine()
    assert_array_equal(aff, hdr.get_base_affine())
    hdr.set_data_shape((3, 5, 7))
    hdr.set_zooms((3, 2, 1))
    assert hdr.default_x_flip
    assert_array_almost_equal(hdr.get_origin_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
    hdr['origin'][:3] = [3, 4, 5]
    assert_array_almost_equal(hdr.get_origin_affine(), [[-3.0, 0.0, 0.0, 6.0], [0.0, 2.0, 0.0, -6.0], [0.0, 0.0, 1.0, -4.0], [0.0, 0.0, 0.0, 1.0]])
    hdr['origin'] = 0
    hdr.set_data_shape((3, 5))
    assert_array_almost_equal(hdr.get_origin_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
    hdr.set_data_shape((3, 5, 7))
    assert_array_almost_equal(hdr.get_origin_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])