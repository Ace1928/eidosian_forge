import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
def test_write_noaffine_mgh():
    v = np.ones((7, 13, 3, 22), np.uint8)
    img = MGHImage(v, None)
    with InTemporaryDirectory():
        save(img, 'tmpsave.mgz')
        mgz = load('tmpsave.mgz')
        h = mgz.header
        del mgz
    assert h['version'] == 1
    assert h['type'] == 0
    assert h['dof'] == 0
    assert h['goodRASFlag'] == 1
    assert np.array_equal(h['dims'], [7, 13, 3, 22])
    assert_almost_equal(h['tr'], 0.0)
    assert_almost_equal(h['flip_angle'], 0.0)
    assert_almost_equal(h['te'], 0.0)
    assert_almost_equal(h['ti'], 0.0)
    assert_almost_equal(h['fov'], 0.0)
    assert_array_almost_equal(h['Mdc'], [[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert_array_almost_equal(h['Pxyz_c'], [0, 0, 0])