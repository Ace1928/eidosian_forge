import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
def _axis_properties(self, axis, key, plot, dimension=None, ax_mapping=None):
    if ax_mapping is None:
        ax_mapping = {'x': 0, 'y': 1}
    axis_props = super()._axis_properties(axis, key, plot, dimension, ax_mapping)
    proj = self.projection
    if self.geographic and proj is GOOGLE_MERCATOR:
        dimension = 'lon' if axis == 'x' else 'lat'
        axis_props['ticker'] = MercatorTicker(dimension=dimension)
        axis_props['formatter'] = MercatorTickFormatter(dimension=dimension)
    return axis_props