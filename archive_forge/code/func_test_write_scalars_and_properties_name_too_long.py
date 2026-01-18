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
def test_write_scalars_and_properties_name_too_long(self):
    for nb_chars in range(22):
        data_per_point = {'A' * nb_chars: DATA['colors']}
        tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        if nb_chars > 18:
            with pytest.raises(ValueError):
                trk.save(BytesIO())
        else:
            trk.save(BytesIO())
        data_per_point = {'A' * nb_chars: DATA['fa']}
        tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        if nb_chars > 20:
            with pytest.raises(ValueError):
                trk.save(BytesIO())
        else:
            trk.save(BytesIO())
    for nb_chars in range(22):
        data_per_streamline = {'A' * nb_chars: DATA['mean_colors']}
        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        if nb_chars > 18:
            with pytest.raises(ValueError):
                trk.save(BytesIO())
        else:
            trk.save(BytesIO())
        data_per_streamline = {'A' * nb_chars: DATA['mean_torsion']}
        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        if nb_chars > 20:
            with pytest.raises(ValueError):
                trk.save(BytesIO())
        else:
            trk.save(BytesIO())