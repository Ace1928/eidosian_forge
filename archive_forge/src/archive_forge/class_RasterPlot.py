import numpy as np
import param
from ...core.options import SkipRendering
from ...core.util import isfinite
from ...element import Image, Raster
from ..mixins import HeatMapMixin
from .element import ColorbarPlot
class RasterPlot(ColorbarPlot):
    nodata = param.Integer(default=None, doc='\n        Optional missing-data value for integer data.\n        If non-None, data with this value will be replaced with NaN so\n        that it is transparent (by default) when plotted.')
    padding = param.ClassSelector(default=0, class_=(int, float, tuple))
    style_opts = ['visible', 'cmap', 'alpha']

    @classmethod
    def trace_kwargs(cls, is_geo=False, **kwargs):
        return {'type': 'heatmap'}

    def graph_options(self, element, ranges, style, **kwargs):
        opts = super().graph_options(element, ranges, style, **kwargs)
        copts = self.get_color_opts(element.vdims[0], element, ranges, style)
        cmin, cmax = (copts.pop('cmin'), copts.pop('cmax'))
        if isfinite(cmin):
            opts['zmin'] = cmin
        if isfinite(cmax):
            opts['zmax'] = cmax
        opts['zauto'] = copts.pop('cauto')
        return dict(opts, **copts)

    def get_data(self, element, ranges, style, **kwargs):
        if isinstance(element, Image):
            l, b, r, t = element.bounds.lbrt()
        else:
            l, b, r, t = element.extents
        array = element.dimension_values(2, flat=False)
        if type(element) is Raster:
            array = array.T[::-1, ...]
        ny, nx = array.shape
        if any((not isfinite(c) for c in (l, b, r, t))) or nx == 0 or ny == 0:
            l, b, r, t, dx, dy = (0, 0, 0, 0, 0, 0)
        else:
            dx, dy = (float(r - l) / nx, float(t - b) / ny)
        x0, y0 = (l + dx / 2.0, b + dy / 2.0)
        if self.invert_axes:
            x0, y0, dx, dy = (y0, x0, dy, dx)
            array = array.T
        return [dict(x0=x0, y0=y0, dx=dx, dy=dy, z=array)]