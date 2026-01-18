import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_out_of_domain_efficiency(self):
    line_string = sgeom.LineString([(0, -90), (2, -90)])
    tgt_proj = ccrs.NorthPolarStereo()
    src_proj = ccrs.PlateCarree()
    cutoff_time = time.time() + 1
    tgt_proj.project_geometry(line_string, src_proj)
    assert time.time() < cutoff_time, 'Projection took too long'