import os
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import data_path, error_warnings
from .. import tck as tck_module
from ..array_sequence import ArraySequence
from ..tck import TckFile
from ..tractogram import Tractogram
from ..tractogram_file import DataError, HeaderError, HeaderWarning
from .test_tractogram import assert_tractogram_equal
def test_write_bigheader_file(self):
    tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    tck_file = BytesIO()
    tck = TckFile(tractogram)
    tck.header['new_entry'] = ' ' * 20
    tck.save(tck_file)
    tck_file.seek(0, os.SEEK_SET)
    new_tck = TckFile.load(tck_file)
    assert_tractogram_equal(new_tck.tractogram, tractogram)
    assert new_tck.header['_offset_data'] == 99
    tck_file = BytesIO()
    tck = TckFile(tractogram)
    tck.header['new_entry'] = ' ' * 21
    tck.save(tck_file)
    tck_file.seek(0, os.SEEK_SET)
    new_tck = TckFile.load(tck_file)
    assert_tractogram_equal(new_tck.tractogram, tractogram)
    assert new_tck.header['_offset_data'] == 101