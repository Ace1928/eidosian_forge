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
def test_write_annot_fill_ctab():
    """Test the `fill_ctab` parameter to :func:`.write_annot`."""
    nvertices = 10
    nlabels = 3
    names = [f'label {l}' for l in range(1, nlabels + 1)]
    labels = list(range(nlabels)) + list(np.random.randint(0, nlabels, nvertices - nlabels))
    labels = np.array(labels, dtype=np.int32)
    np.random.shuffle(labels)
    rgba = np.array(np.random.randint(0, 255, (nlabels, 4)), dtype=np.int32)
    annot_path = 'c.annot'
    with InTemporaryDirectory():
        write_annot(annot_path, labels, rgba, names, fill_ctab=True)
        labels2, rgbal2, names2 = read_annot(annot_path)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, labels))
        assert names2 == names
        badannot = (10 * np.arange(nlabels, dtype=np.int32)).reshape(-1, 1)
        rgbal = np.hstack((rgba, badannot))
        with pytest.warns(UserWarning, match=f'Annotation values in {annot_path} will be incorrect'):
            write_annot(annot_path, labels, rgbal, names, fill_ctab=False)
        labels2, rgbal2, names2 = read_annot(annot_path, orig_ids=True)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, badannot[labels].squeeze()))
        assert names2 == names
        rgbal = np.hstack((rgba, np.zeros((nlabels, 1), dtype=np.int32)))
        rgbal[:, 4] = rgbal[:, 0] + rgbal[:, 1] * 2 ** 8 + rgbal[:, 2] * 2 ** 16
        with clear_and_catch_warnings() as w:
            write_annot(annot_path, labels, rgbal, names, fill_ctab=False)
        assert all((f'Annotation values in {annot_path} will be incorrect' != str(ww.message) for ww in w))
        labels2, rgbal2, names2 = read_annot(annot_path)
        names2 = [n.decode('ascii') for n in names2]
        assert np.all(np.isclose(rgbal2[:, :4], rgba))
        assert np.all(np.isclose(labels2, labels))
        assert names2 == names