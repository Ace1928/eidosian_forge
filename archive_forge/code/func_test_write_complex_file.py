import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
def test_write_complex_file(self):
    tractogram = Tractogram(DATA['streamlines'], data_per_point=DATA['data_per_point'], affine_to_rasmm=np.eye(4))
    trk_file = BytesIO()
    trk = TrkFile(tractogram)
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file, lazy_load=False)
    assert_tractogram_equal(new_trk.tractogram, tractogram)
    data_per_streamline = DATA['data_per_streamline']
    tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
    trk = TrkFile(tractogram)
    trk_file = BytesIO()
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file, lazy_load=False)
    assert_tractogram_equal(new_trk.tractogram, tractogram)
    data_per_streamline = DATA['data_per_streamline']
    tractogram = Tractogram(DATA['streamlines'], data_per_point=DATA['data_per_point'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
    trk_file = BytesIO()
    trk = TrkFile(tractogram)
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file, lazy_load=False)
    assert_tractogram_equal(new_trk.tractogram, tractogram)
    new_trk_orig = TrkFile.load(DATA['complex_trk_fname'])
    assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
    trk_file.seek(0, os.SEEK_SET)
    assert trk_file.read() == open(DATA['complex_trk_fname'], 'rb').read()