import pathlib
import pickle
import warnings
from pickle import dumps, HIGHEST_PROTOCOL, loads
import pytest
import shapely
from shapely import wkt
from shapely.geometry import (
@pytest.mark.parametrize('fname', (HERE / 'data').glob('*.pickle'), ids=lambda fname: fname.name)
def test_unpickle_pre_20(fname):
    from shapely.testing import assert_geometries_equal
    geom_type = fname.name.split('_')[0]
    expected = TEST_DATA[geom_type]
    with open(fname, 'rb') as f:
        with pytest.warns(UserWarning):
            result = pickle.load(f)
    assert_geometries_equal(result, expected)