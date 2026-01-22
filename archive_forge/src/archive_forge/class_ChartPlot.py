import numpy as np
import param
from ...element import Tiles
from ...operation import interpolate_curve
from ..mixins import AreaMixin, BarsMixin
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class ChartPlot(ElementPlot):

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'scatter'}

    def get_data(self, element, ranges, style, is_geo=False, **kwargs):
        if is_geo:
            if self.invert_axes:
                x = element.dimension_values(1)
                y = element.dimension_values(0)
            else:
                x = element.dimension_values(0)
                y = element.dimension_values(1)
            lon, lat = Tiles.easting_northing_to_lon_lat(x, y)
            return [{'lon': lon, 'lat': lat}]
        else:
            x, y = ('y', 'x') if self.invert_axes else ('x', 'y')
            return [{x: element.dimension_values(0), y: element.dimension_values(1)}]