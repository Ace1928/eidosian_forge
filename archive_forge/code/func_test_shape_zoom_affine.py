import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_shape_zoom_affine():
    shape = (3, 5, 7)
    zooms = (3, 2, 1)
    res = shape_zoom_affine(shape, zooms)
    exp = np.array([[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine((3, 5), (3, 2))
    exp = np.array([[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine(shape, zooms, False)
    exp = np.array([[3.0, 0.0, 0.0, -3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
    assert_array_almost_equal(res, exp)