import os
import tempfile
import unittest
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.compat.py3k import asbytes
import nibabel as nib
from nibabel.testing import clear_and_catch_warnings, data_path, error_warnings
from nibabel.tmpdirs import InTemporaryDirectory
from .. import FORMATS, trk
from ..tractogram import LazyTractogram, Tractogram
from ..tractogram_file import ExtensionWarning, TractogramFile
from .test_tractogram import assert_tractogram_equal
def test_save_complex_file(self):
    complex_tractogram = Tractogram(DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'], affine_to_rasmm=np.eye(4))
    for ext, cls in FORMATS.items():
        with InTemporaryDirectory():
            filename = 'streamlines' + ext
            nb_expected_warnings = (not cls.SUPPORTS_DATA_PER_POINT) + (not cls.SUPPORTS_DATA_PER_STREAMLINE)
            with clear_and_catch_warnings() as w:
                warnings.simplefilter('always')
                nib.streamlines.save(complex_tractogram, filename)
            assert len(w) == nb_expected_warnings
            tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
            if cls.SUPPORTS_DATA_PER_POINT:
                tractogram.data_per_point = DATA['data_per_point']
            if cls.SUPPORTS_DATA_PER_STREAMLINE:
                data = DATA['data_per_streamline']
                tractogram.data_per_streamline = data
            tfile = nib.streamlines.load(filename, lazy_load=False)
            assert_tractogram_equal(tfile.tractogram, tractogram)