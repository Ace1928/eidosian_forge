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
def test_reshaped_is_proxy():
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert isinstance(prox.reshape((2, 3, 4)), ArrayProxy)
    minus1 = prox.reshape((2, -1, 4))
    assert isinstance(minus1, ArrayProxy)
    assert minus1.shape == (2, 3, 4)
    with pytest.raises(ValueError):
        prox.reshape((-1, -1, 4))
    with pytest.raises(ValueError):
        prox.reshape((2, 3, 5))
    with pytest.raises(ValueError):
        prox.reshape((2, -1, 5))