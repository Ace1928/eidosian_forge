import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_simple_run_out(self):
    projection = FakeProjection(left_offset=10)
    line_string = sgeom.LineString([(-175, 0), (-160, 0)])
    multi_line_string = projection.project_geometry(line_string)
    assert len(multi_line_string.geoms) == 1
    assert len(multi_line_string.geoms[0].coords) == 2