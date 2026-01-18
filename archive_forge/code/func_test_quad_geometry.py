import getpass
import hashlib
import os
import struct
import time
import unittest
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...fileslice import strided_scalar
from ...testing import clear_and_catch_warnings
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...tmpdirs import InTemporaryDirectory
from .. import (
from ..io import _pack_rgb
@freesurfer_test
@needs_nibabel_data('nitest-freesurfer')
def test_quad_geometry():
    """Test IO of freesurfer quad files."""
    new_quad = pjoin(get_nibabel_data(), 'nitest-freesurfer', 'subjects', 'bert', 'surf', 'lh.inflated.nofix')
    coords, faces = read_geometry(new_quad)
    assert 0 == faces.min()
    assert coords.shape[0] == faces.max() + 1
    with InTemporaryDirectory():
        new_path = 'test'
        write_geometry(new_path, coords, faces)
        coords2, faces2 = read_geometry(new_path)
        assert np.array_equal(coords, coords2)
        assert np.array_equal(faces, faces2)