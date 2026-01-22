import itertools
from collections import defaultdict
from html import escape
import numpy as np
import pandas as pd
import param
from bokeh.models import Arrow, BoxAnnotation, NormalHead, Slope, Span, TeeHead
from bokeh.transform import dodge
from panel.models import HTML
from ...core.util import datetime_types, dimension_sanitizer
from ...element import HLine, HLines, HSpans, VLine, VLines, VSpan, VSpans
from ..plot import GenericElementPlot
from .element import AnnotationPlot, ColorbarPlot, CompositeElementPlot, ElementPlot
from .plot import BokehPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
from .util import bokeh32, date_to_integer
class LabelsPlot(ColorbarPlot, AnnotationPlot):
    show_legend = param.Boolean(default=False, doc='\n        Whether to show legend for the plot.')
    xoffset = param.Number(default=None, doc='\n      Amount of offset to apply to labels along x-axis.')
    yoffset = param.Number(default=None, doc='\n      Amount of offset to apply to labels along x-axis.')
    color_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc="\n        Deprecated in favor of color style mapping, e.g. `color=dim('color')`")
    selection_display = BokehOverlaySelectionDisplay()
    style_opts = base_properties + text_properties + ['cmap', 'angle']
    _nonvectorized_styles = base_properties + ['cmap']
    _plot_methods = dict(single='text', batched='text')
    _batched_style_opts = text_properties

    def get_data(self, element, ranges, style):
        style = self.style[self.cyclic_index]
        if 'angle' in style and isinstance(style['angle'], (int, float)):
            style['angle'] = np.deg2rad(style.get('angle', 0))
        dims = element.dimensions()
        coords = (1, 0) if self.invert_axes else (0, 1)
        xdim, ydim, tdim = (dimension_sanitizer(dims[i].name) for i in coords + (2,))
        mapping = dict(x=xdim, y=ydim, text=tdim)
        data = {d: element.dimension_values(d) for d in (xdim, ydim)}
        if self.xoffset is not None:
            mapping['x'] = dodge(xdim, self.xoffset)
        if self.yoffset is not None:
            mapping['y'] = dodge(ydim, self.yoffset)
        data[tdim] = [dims[2].pprint_value(v) for v in element.dimension_values(2)]
        self._categorize_data(data, (xdim, ydim), element.dimensions())
        cdim = element.get_dimension(self.color_index)
        if cdim is None:
            return (data, mapping, style)
        cdata, cmapping = self._get_color_data(element, ranges, style, name='text_color')
        if dims[2] is cdim and cdata:
            data['text_color'] = cdata[tdim]
            mapping['text_color'] = dict(cmapping['text_color'], field='text_color')
        else:
            data.update(cdata)
            mapping.update(cmapping)
        return (data, mapping, style)