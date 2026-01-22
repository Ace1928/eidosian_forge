import numpy as np
import param
from ...element import HLine, HSpan, Tiles, VLine, VSpan
from ..mixins import GeomMixin
from .element import ElementPlot
class BoxShapePlot(GeomMixin, ShapePlot):
    _shape_type = 'rect'

    def get_data(self, element, ranges, style, is_geo=False, **kwargs):
        inds = (1, 0, 3, 2) if self.invert_axes else (0, 1, 2, 3)
        x0s, y0s, x1s, y1s = (element.dimension_values(kd) for kd in inds)
        if is_geo:
            if len(x0s) == 0:
                lat = []
                lon = []
            else:
                lon0s, lat0s = Tiles.easting_northing_to_lon_lat(easting=x0s, northing=y0s)
                lon1s, lat1s = Tiles.easting_northing_to_lon_lat(easting=x1s, northing=y1s)
                lon_chunks, lat_chunks = zip(*[([lon0, lon0, lon1, lon1, lon0, np.nan], [lat0, lat1, lat1, lat0, lat0, np.nan]) for lon0, lat0, lon1, lat1 in zip(lon0s, lat0s, lon1s, lat1s)])
                lon = np.concatenate(lon_chunks)
                lat = np.concatenate(lat_chunks)
            return [{'lat': lat, 'lon': lon}]
        else:
            return [dict(x0=x0, x1=x1, y0=y0, y1=y1, xref='x', yref='y') for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s)]