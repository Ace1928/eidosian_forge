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
Assert that array proxies return memory maps as expected

    Parameters
    ----------
    hdr : object
        Image header instance
    offset : int
        Offset in bytes of image data in file (that we will write)
    proxy_class : class
        Class of image array proxy to test
    has_scaling : {False, True}
        True if the `hdr` says to apply scaling to the output data, False
        otherwise.
    unscaled_is_view : {True, False}
        True if getting the unscaled data returns a view of the array.  If
        False, then type of returned array will depend on whether numpy has the
        old viral (< 1.12) memmap behavior (returns memmap) or the new behavior
        (returns ndarray).  See: https://github.com/numpy/numpy/pull/7406
    