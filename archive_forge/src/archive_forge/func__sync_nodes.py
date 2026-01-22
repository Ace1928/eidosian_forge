import numpy as np
import param
from bokeh.models import Patches
from ...core.data import Dataset
from ...core.util import dimension_sanitizer, max_range
from ...util.transform import dim
from .graphs import GraphPlot
def _sync_nodes(self):
    arc_renderer = self.handles['quad_1_glyph_renderer']
    scatter_renderer = self.handles['scatter_1_glyph_renderer']
    for gtype in ('selection_', 'nonselection_', 'muted_', 'hover_', ''):
        glyph = getattr(scatter_renderer, gtype + 'glyph')
        arc_glyph = getattr(arc_renderer, gtype + 'glyph')
        if not glyph or not arc_glyph:
            continue
        scatter_props = glyph.properties_with_values(include_defaults=False)
        styles = {k: v for k, v in scatter_props.items() if k in arc_glyph.properties()}
        arc_glyph.update(**styles)