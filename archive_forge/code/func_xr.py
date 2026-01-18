import pytest
import sys
import matplotlib
from matplotlib import _api
@pytest.fixture
def xr():
    """Fixture to import xarray."""
    xr = pytest.importorskip('xarray')
    return xr