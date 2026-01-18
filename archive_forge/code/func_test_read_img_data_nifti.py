import pathlib
import shutil
from os.path import dirname
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import numpy as np
from .. import (
from ..filebasedimages import ImageFileError
from ..loadsave import _signature_matches_extension, load, read_img_data
from ..openers import Opener
from ..optpkg import optional_package
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
@expires('5.0.0')
def test_read_img_data_nifti():
    shape = (2, 3, 4)
    data = np.random.normal(size=shape)
    out_dtype = np.dtype(np.int16)
    classes = (Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image)
    if have_scipy:
        classes += (Spm99AnalyzeImage, Spm2AnalyzeImage)
    with InTemporaryDirectory():
        for i, img_class in enumerate(classes):
            img = img_class(data, np.eye(4))
            img.set_data_dtype(out_dtype)
            with deprecated_to('5.0.0'), pytest.raises(ImageFileError):
                read_img_data(img)
            froot = f'an_image_{i}'
            img.file_map = img.filespec_to_file_map(froot)
            with deprecated_to('5.0.0'), pytest.raises(OSError):
                read_img_data(img)
            img.to_file_map()
            img_fname = img.file_map['image'].filename
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            hdr_fname = img.file_map['header'].filename if 'header' in img.file_map else img_fname
            with open(hdr_fname, 'rb') as fobj:
                hdr_back = img_back.header_class.from_fileobj(fobj)
            with open(img_fname, 'rb') as fobj:
                scaled_back = hdr_back.data_from_fileobj(fobj)
            assert_array_equal(data_back, scaled_back)
            with open(img_fname, 'rb') as fobj:
                unscaled_back = hdr_back.raw_data_from_fileobj(fobj)
            with deprecated_to('5.0.0'):
                assert_array_equal(unscaled_back, read_img_data(img_back, prefer='unscaled'))
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            has_inter = hdr_back.has_data_intercept
            old_slope = hdr_back['scl_slope']
            old_inter = hdr_back['scl_inter'] if has_inter else 0
            est_unscaled = (data_back - old_inter) / old_slope
            with deprecated_to('5.0.0'):
                actual_unscaled = read_img_data(img_back, prefer='unscaled')
            assert_almost_equal(est_unscaled, actual_unscaled)
            img_back.header['scl_slope'] = 2.1
            if has_inter:
                new_inter = 3.14
                img_back.header['scl_inter'] = 3.14
            else:
                new_inter = 0
            with deprecated_to('5.0.0'):
                assert np.allclose(actual_unscaled * 2.1 + new_inter, read_img_data(img_back))
            with deprecated_to('5.0.0'):
                assert_array_equal(actual_unscaled, read_img_data(img_back, prefer='unscaled'))
            img.header.set_data_offset(1024)
            del actual_unscaled, unscaled_back
            img.to_file_map()
            with open(img_fname, 'ab') as fobj:
                fobj.write(b'\x00\x00')
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with deprecated_to('5.0.0'):
                assert_array_equal(data_back, read_img_data(img_back))
            img_back.header.set_data_offset(1026)
            exp_offset = np.zeros((data.size,), data.dtype) + old_inter
            exp_offset[:-1] = np.ravel(data_back, order='F')[1:]
            exp_offset = np.reshape(exp_offset, shape, order='F')
            with deprecated_to('5.0.0'):
                assert_array_equal(exp_offset, read_img_data(img_back))
            del img, img_back, data_back