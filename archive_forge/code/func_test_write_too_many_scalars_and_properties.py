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
def test_write_too_many_scalars_and_properties(self):
    data_per_point = {}
    for i in range(10):
        data_per_point[f'#{i}'] = DATA['fa']
        tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
    data_per_point[f'#{i + 1}'] = DATA['fa']
    tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
    trk = TrkFile(tractogram)
    with pytest.raises(ValueError):
        trk.save(BytesIO())
    data_per_streamline = {}
    for i in range(10):
        data_per_streamline[f'#{i}'] = DATA['mean_torsion']
        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
    data_per_streamline[f'#{i + 1}'] = DATA['mean_torsion']
    tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline)
    trk = TrkFile(tractogram)
    with pytest.raises(ValueError):
        trk.save(BytesIO())