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
def test_tractogram_file_properties(self):
    trk = TrkFile.load(DATA['simple_trk_fname'])
    assert trk.streamlines == trk.tractogram.streamlines
    assert_array_equal(trk.affine, trk.header[Field.VOXEL_TO_RASMM])