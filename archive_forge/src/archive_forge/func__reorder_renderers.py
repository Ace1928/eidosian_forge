from collections import defaultdict
import numpy as np
import param
from bokeh.models import (
from ...core.data import Dataset
from ...core.options import Cycle, abbreviated_exception
from ...core.util import dimension_sanitizer, unique_array
from ...util.transform import dim
from ..mixins import ChordMixin, GraphMixin
from ..util import get_directed_graph_paths, process_cmap
from .chart import ColorbarPlot, PointPlot
from .element import CompositeElementPlot, LegendPlot
from .styles import (
def _reorder_renderers(self, plot, renderer, mapping):
    """Reorders renderers based on the defined draw order"""
    renderers = dict({r: self.handles[r + '_glyph_renderer'] for r in mapping}, graph=renderer)
    other = [r for r in plot.renderers if r not in renderers.values()]
    graph_renderers = [renderers[k] for k in self._draw_order if k in renderers]
    plot.renderers = other + graph_renderers