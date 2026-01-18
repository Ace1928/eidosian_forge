import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_xyzservices_query_name(self):
    pytest.importorskip('xyzservices')
    m = self.nybb.explore(tiles='CartoDB Positron No Labels')
    out_str = self._fetch_map_string(m)
    assert '"https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"' in out_str
    assert 'attribution":"\\u0026copy;\\u003cahref=\\"https://www.openstreetmap.org' in out_str
    assert '"maxNativeZoom":20,"maxZoom":20,"minZoom":0' in out_str