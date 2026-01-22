import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
def _update_hover(self, element):
    tooltips, hover_opts = self._hover_opts(element)
    hover = self.handles['hover']
    if 'hv_created' in hover.tags:
        tooltips = [(ttp.pprint_label, '@{%s}' % dimension_sanitizer(ttp.name)) if isinstance(ttp, Dimension) else ttp for ttp in tooltips]
        if self.geographic and tooltips[2:] == hover.tooltips[2:]:
            return
        tooltips = [(l, t + '{custom}' if t in hover.formatters else t) for l, t in tooltips]
        hover.tooltips = tooltips
    else:
        super()._update_hover(element)