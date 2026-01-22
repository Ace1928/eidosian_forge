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
class LineAnnotationPlot(ElementPlot, AnnotationPlot):
    style_opts = line_properties + ['level', 'visible']
    apply_ranges = param.Boolean(default=False, doc='\n        Whether to include the annotation in axis range calculations.')
    _allow_implicit_categories = False
    _plot_methods = dict(single='Span')
    selection_display = None

    def get_data(self, element, ranges, style):
        data, mapping = ({}, {})
        dim = 'width' if isinstance(element, HLine) else 'height'
        if self.invert_axes:
            dim = 'width' if dim == 'height' else 'height'
        mapping['dimension'] = dim
        loc = element.data
        if isinstance(loc, datetime_types):
            loc = date_to_integer(loc)
        mapping['location'] = loc
        return (data, mapping, style)

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        box = Span(level=properties.get('level', 'glyph'), **mapping)
        plot.renderers.append(box)
        return (None, box)

    def get_extents(self, element, ranges=None, range_type='combined', **kwargs):
        loc = element.data
        if isinstance(element, VLine):
            dim = 'x'
        elif isinstance(element, HLine):
            dim = 'y'
        if self.invert_axes:
            dim = 'x' if dim == 'y' else 'x'
        ranges[dim]['soft'] = (loc, loc)
        return super().get_extents(element, ranges, range_type)