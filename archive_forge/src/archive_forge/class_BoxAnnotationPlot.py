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
class BoxAnnotationPlot(ElementPlot, AnnotationPlot):
    apply_ranges = param.Boolean(default=False, doc='\n        Whether to include the annotation in axis range calculations.')
    style_opts = line_properties + fill_properties + ['level', 'visible']
    _allow_implicit_categories = False
    _plot_methods = dict(single='BoxAnnotation')
    selection_display = None

    def get_data(self, element, ranges, style):
        data = {}
        mapping = {k: None for k in ('left', 'right', 'bottom', 'top')}
        kwd_dim1 = 'left' if isinstance(element, VSpan) else 'bottom'
        kwd_dim2 = 'right' if isinstance(element, VSpan) else 'top'
        if self.invert_axes:
            kwd_dim1 = 'bottom' if kwd_dim1 == 'left' else 'left'
            kwd_dim2 = 'top' if kwd_dim2 == 'right' else 'right'
        locs = element.data
        if isinstance(locs, datetime_types):
            locs = [date_to_integer(loc) for loc in locs]
        mapping[kwd_dim1] = locs[0]
        mapping[kwd_dim2] = locs[1]
        return (data, mapping, style)

    def _update_glyph(self, renderer, properties, mapping, glyph, source, data):
        glyph.visible = any((v is not None for v in mapping.values()))
        return super()._update_glyph(renderer, properties, mapping, glyph, source, data)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        box = BoxAnnotation(level=properties.get('level', 'glyph'), **mapping)
        plot.renderers.append(box)
        return (None, box)