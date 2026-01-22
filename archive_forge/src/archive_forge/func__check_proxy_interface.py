import io
import pathlib
import sys
import warnings
from functools import partial
from itertools import product
import numpy as np
from ..optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from nibabel.arraywriters import WriterError
from nibabel.testing import (
from .. import (
from ..casting import sctypes
from ..spatialimages import SpatialImage
from ..tmpdirs import InTemporaryDirectory
from .test_api_validators import ValidateAPI
from .test_brikhead import EXAMPLE_IMAGES as AFNI_EXAMPLE_IMAGES
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES
def _check_proxy_interface(self, imaker, meth_name):
    img = imaker()
    assert is_proxy(img.dataobj)
    assert not isinstance(img.dataobj, np.ndarray)
    proxy_data = np.asarray(img.dataobj)
    proxy_copy = proxy_data.copy()
    assert not img.in_memory
    method = getattr(img, meth_name)
    data = method(caching='unchanged')
    assert not img.in_memory
    data = method()
    assert img.in_memory
    assert not proxy_data is data
    assert_array_equal(proxy_data, data)
    data_again = method(caching='unchanged')
    assert data is data_again
    data_yet_again = method(caching='fill')
    assert data is data_yet_again
    data[:] = 42
    assert_array_equal(proxy_data, proxy_copy)
    assert_array_equal(np.asarray(img.dataobj), proxy_copy)
    assert_array_equal(method(), 42)
    img.uncache()
    assert not img.in_memory
    assert_array_equal(method(), proxy_copy)
    img = imaker()
    method = getattr(img, meth_name)
    assert not img.in_memory
    data = method(caching='fill')
    assert img.in_memory
    data_again = method()
    assert data is data_again
    img.uncache()
    fdata = img.get_fdata()
    assert fdata.dtype == np.float64
    fdata[:] = 42
    fdata_back = img.get_fdata()
    assert_array_equal(fdata_back, 42)
    assert fdata_back.dtype == np.float64
    fdata_new_dt = img.get_fdata(caching='unchanged', dtype='f4')
    assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
    assert fdata_new_dt.dtype == np.float32
    assert_array_equal(img.get_fdata(), 42)
    fdata_new_dt[:] = 43
    fdata_new_dt = img.get_fdata(caching='unchanged', dtype='f4')
    assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
    fdata_new_dt = img.get_fdata(caching='fill', dtype='f4')
    assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
    fdata_new_dt[:] = 43
    assert_array_equal(img.get_fdata(dtype='f4'), 43)
    assert_array_equal(img.get_fdata(), proxy_data)