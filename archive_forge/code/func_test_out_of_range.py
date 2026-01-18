import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
def test_out_of_range(self, Source):
    source = Source()
    match = 'No srtm tile found for those coordinates\\.'
    with pytest.raises(ValueError, match=match):
        source.single_tile(-25, 50)