import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
def _postprocess_hover(self, renderer, source):
    super()._postprocess_hover(renderer, source)
    hover = self.handles['plot'].hover
    hover = hover[0] if hover else None
    if not self.geographic or hover is None or isinstance(hover.tooltips, str) or (self.projection is not GOOGLE_MERCATOR) or (hover.tooltips is None) or ('hv_created' not in hover.tags):
        return
    element = self.current_frame
    xdim, ydim = (dimension_sanitizer(kd.name) for kd in element.kdims)
    formatters, tooltips = (dict(hover.formatters), [])
    xhover = CustomJSHover(code=self._hover_code % 0)
    yhover = CustomJSHover(code=self._hover_code % 1)
    for name, formatter in hover.tooltips:
        customjs = None
        if formatter in ('@{%s}' % xdim, '$x'):
            dim = xdim
            formatter = '$x'
            customjs = xhover
        elif formatter in ('@{%s}' % ydim, '$y'):
            dim = ydim
            formatter = '$y'
            customjs = yhover
        if customjs:
            key = formatter if formatter in ('$x', '$y') else dim
            formatters[key] = customjs
            formatter += '{custom}'
        tooltips.append((name, formatter))
    hover.tooltips = tooltips
    hover.formatters = formatters