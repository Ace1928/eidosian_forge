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
def test_reshape_dataobj():
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    arr = np.arange(np.prod(shape), dtype=prox.dtype).reshape(shape)
    bio.write(b'\x00' * prox.offset + arr.tobytes(order='F'))
    assert_array_equal(prox, arr)
    assert_array_equal(reshape_dataobj(prox, (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert prox.shape == shape
    assert arr.shape == shape
    assert_array_equal(reshape_dataobj(arr, (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert arr.shape == shape

    class ArrGiver:

        def __array__(self):
            return arr
    assert_array_equal(reshape_dataobj(ArrGiver(), (2, 3, 4)), np.reshape(arr, (2, 3, 4)))
    assert arr.shape == shape