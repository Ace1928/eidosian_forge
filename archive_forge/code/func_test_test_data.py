import os
import sys
import warnings
import numpy as np
import pytest
from ..casting import sctypes
from ..testing import (
def test_test_data():
    assert str(get_test_data()) == str(data_path)
    assert str(get_test_data()) == os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data'))
    for subdir in ('nicom', 'gifti', 'externals'):
        assert get_test_data(subdir) == data_path.parent.parent / subdir / 'tests' / 'data'
        assert os.path.exists(get_test_data(subdir))
        assert not os.path.exists(get_test_data(subdir, 'doesnotexist'))
    for subdir in ('freesurfer', 'doesnotexist'):
        with pytest.raises(ValueError):
            get_test_data(subdir)
    assert not os.path.exists(get_test_data(None, 'doesnotexist'))
    for subdir, fname in [('gifti', 'ascii.gii'), ('nicom', '0.dcm'), ('externals', 'example_1.nc'), (None, 'empty.tck')]:
        assert os.path.exists(get_test_data(subdir, fname))