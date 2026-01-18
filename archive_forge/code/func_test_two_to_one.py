import logging
import pathlib
import shutil
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from tempfile import mkdtemp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import (
from .. import analyze as ana
from .. import loadsave as nils
from .. import nifti1 as ni1
from .. import spm2analyze as spm2
from .. import spm99analyze as spm99
from ..optpkg import optional_package
from ..spatialimages import SpatialImage
from ..testing import deprecated_to, expires
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import native_code, swapped_code
def test_two_to_one():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3, 3] = [3, 2, 1]
    img = ni1.Nifti1Image(data, affine)
    assert img.header['magic'] == b'n+1'
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    assert img.header['magic'] == b'n+1'
    assert img.header['vox_offset'] == 0
    pimg = ni1.Nifti1Pair(data, affine, img.header)
    isio = BytesIO()
    hsio = BytesIO()
    pimg.file_map['image'].fileobj = isio
    pimg.file_map['header'].fileobj = hsio
    pimg.to_file_map()
    assert pimg.header['magic'] == b'ni1'
    assert pimg.header['vox_offset'] == 0
    assert_array_equal(pimg.get_fdata(), data)
    ana_img = ana.AnalyzeImage.from_image(img)
    assert ana_img.header['vox_offset'] == 0
    str_io = BytesIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    assert img.header['vox_offset'] == 0
    aimg = ana.AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    aimg = spm99.Spm99AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    aimg = spm2.Spm2AnalyzeImage.from_image(img)
    assert aimg.header['vox_offset'] == 0
    nfimg = ni1.Nifti1Pair.from_image(img)
    assert nfimg.header['vox_offset'] == 0
    hdr = nfimg.header
    hdr['vox_offset'] = 16
    assert nfimg.header['vox_offset'] == 16
    nfimg = ni1.Nifti1Image.from_image(img)
    assert nfimg.header['vox_offset'] == 0