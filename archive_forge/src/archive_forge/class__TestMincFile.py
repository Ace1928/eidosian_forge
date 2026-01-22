import bz2
import gzip
import types
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, load, minc1
from ..deprecated import ModuleProxy
from ..deprecator import ExpiredDeprecationError
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image, MincHeader
from ..optpkg import optional_package
from ..testing import assert_data_similar, clear_and_catch_warnings, data_path
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples
class _TestMincFile:
    module = minc1
    file_class = Minc1File
    fname = EG_FNAME
    opener = netcdf_file
    test_files = EXAMPLE_IMAGES

    def test_mincfile(self):
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            assert mnc.get_data_dtype().type == tp['dtype']
            assert mnc.get_data_shape() == tp['shape']
            assert mnc.get_zooms() == tp['zooms']
            assert_array_equal(mnc.get_affine(), tp['affine'])
            data = mnc.get_scaled_data()
            assert data.shape == tp['shape']
            del mnc, data

    def test_mincfile_slicing(self):
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            data = mnc.get_scaled_data()
            for slicedef in ((slice(None),), (1,), (slice(None), 1), (1, slice(None)), (slice(None), 1, 1), (1, slice(None), 1), (1, 1, slice(None))):
                sliced_data = mnc.get_scaled_data(slicedef)
                assert_array_equal(sliced_data, data[slicedef])
            del mnc, data

    def test_load(self):
        for tp in self.test_files:
            img = load(tp['fname'])
            data = img.get_fdata()
            assert data.shape == tp['shape']
            assert_data_similar(data, tp)
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_fdata(), data)

    def test_array_proxy_slicing(self):
        for tp in self.test_files:
            img = load(tp['fname'])
            arr = img.get_fdata()
            prox = img.dataobj
            assert prox.is_proxy
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])