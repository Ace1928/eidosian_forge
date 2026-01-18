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
def test_read_annot_old_format():
    """Test reading an old-style .annot file."""

    def gen_old_annot_file(fpath, nverts, labels, rgba, names):
        dt = '>i'
        vdata = np.zeros((nverts, 2), dtype=dt)
        vdata[:, 0] = np.arange(nverts)
        vdata[:, [1]] = _pack_rgb(rgba[labels, :3])
        fbytes = b''
        fbytes += struct.pack(dt, nverts)
        fbytes += vdata.astype(dt).tobytes()
        fbytes += struct.pack(dt, 1)
        fbytes += struct.pack(dt, rgba.shape[0])
        fbytes += struct.pack(dt, 5)
        fbytes += b'abcd\x00'
        for i in range(rgba.shape[0]):
            fbytes += struct.pack(dt, len(names[i]) + 1)
            fbytes += names[i].encode('ascii') + b'\x00'
            fbytes += rgba[i, :].astype(dt).tobytes()
        with open(fpath, 'wb') as f:
            f.write(fbytes)
    with InTemporaryDirectory():
        nverts = 10
        nlabels = 3
        names = [f'Label {l}' for l in range(nlabels)]
        labels = np.concatenate((np.arange(nlabels), np.random.randint(0, nlabels, nverts - nlabels)))
        np.random.shuffle(labels)
        rgba = np.random.randint(0, 255, (nlabels, 4))
        gen_old_annot_file('blah.annot', nverts, labels, rgba, names)
        rlabels, rrgba, rnames = read_annot('blah.annot')
        rnames = [n.decode('ascii') for n in rnames]
        assert np.all(np.isclose(labels, rlabels))
        assert np.all(np.isclose(rgba, rrgba[:, :4]))
        assert names == rnames