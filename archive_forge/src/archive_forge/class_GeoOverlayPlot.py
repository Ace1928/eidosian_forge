import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
class GeoOverlayPlot(GeoPlot, HvOverlayPlot):
    """
    Subclasses the HoloViews OverlayPlot to add custom behavior
    for geographic plots.
    """
    global_extent = param.Boolean(default=False, doc='\n        Whether the plot should display the whole globe.')
    _propagate_options = HvOverlayPlot._propagate_options + ['global_extent', 'show_bounds', 'infer_projection']

    def __init__(self, element, **params):
        super().__init__(element, **params)
        self.geographic = any(element.traverse(is_geographic, [_Element]))
        if self.geographic:
            self.show_grid = False