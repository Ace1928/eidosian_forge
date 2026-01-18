import contextlib
import gzip
import pickle
from io import BytesIO
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from .. import __version__
from ..arrayproxy import ArrayProxy, get_obj_dtype, is_proxy, reshape_dataobj
from ..deprecator import ExpiredDeprecationError
from ..nifti1 import Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..testing import memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip
@pytest.mark.parametrize('n_dim', (1, 2, 3))
@pytest.mark.parametrize('offset', (0, 20))
def test_proxy_slicing(n_dim, offset):
    shape = (15, 16, 17)[:n_dim]
    arr = np.arange(np.prod(shape)).reshape(shape)
    hdr = Nifti1Header()
    hdr.set_data_offset(offset)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_data_shape(shape)
    for order, klass in (('F', ArrayProxy), ('C', CArrayProxy)):
        fobj = BytesIO()
        fobj.write(b'\x00' * offset)
        fobj.write(arr.tobytes(order=order))
        prox = klass(fobj, hdr)
        assert prox.order == order
        for sliceobj in slicer_samples(shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])