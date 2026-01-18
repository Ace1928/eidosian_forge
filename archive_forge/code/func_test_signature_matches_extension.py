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
def test_signature_matches_extension(tmp_path):
    gz_signature = b'\x1f\x8b'
    good_file = tmp_path / 'good.gz'
    good_file.write_bytes(gz_signature)
    bad_file = tmp_path / 'bad.gz'
    bad_file.write_bytes(b'bad')
    matches, msg = _signature_matches_extension(tmp_path / 'uncompressed.nii')
    assert matches
    assert msg == ''
    matches, msg = _signature_matches_extension(tmp_path / 'missing.gz')
    assert not matches
    assert msg.startswith('Could not read')
    matches, msg = _signature_matches_extension(bad_file)
    assert not matches
    assert 'is not a' in msg
    matches, msg = _signature_matches_extension(good_file)
    assert matches
    assert msg == ''
    matches, msg = _signature_matches_extension(tmp_path / 'missing.nii')
    assert matches
    assert msg == ''