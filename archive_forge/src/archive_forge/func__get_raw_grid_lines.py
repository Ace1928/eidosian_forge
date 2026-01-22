import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
def _get_raw_grid_lines(self, lon_values, lat_values, lon_min, lon_max, lat_min, lat_max):
    lons_i = np.linspace(lon_min, lon_max, 100)
    lats_i = np.linspace(lat_min, lat_max, 100)
    lon_lines = [self.transform_xy(np.full_like(lats_i, lon), lats_i) for lon in lon_values]
    lat_lines = [self.transform_xy(lons_i, np.full_like(lons_i, lat)) for lat in lat_values]
    return (lon_lines, lat_lines)