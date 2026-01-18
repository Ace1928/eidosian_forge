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
def test_write_optional_header_fields(self):
    tractogram = Tractogram(affine_to_rasmm=np.eye(4))
    trk_file = BytesIO()
    header = {'extra': 1234}
    trk = TrkFile(tractogram, header)
    trk.save(trk_file)
    trk_file.seek(0, os.SEEK_SET)
    new_trk = TrkFile.load(trk_file)
    assert 'extra' not in new_trk.header