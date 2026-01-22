from __future__ import annotations
import logging # isort:skip
from ...core.properties import Instance, InstanceDefault
from ...core.validation import error
from ...core.validation.errors import MALFORMED_GRAPH_SOURCE
from ..glyphs import MultiLine, Scatter
from ..graphs import GraphHitTestPolicy, LayoutProvider, NodesOnly
from ..sources import ColumnDataSource
from .glyph_renderer import GlyphRenderer
from .renderer import DataRenderer
@error(MALFORMED_GRAPH_SOURCE)
def _check_malformed_graph_source(self):
    missing = []
    if 'index' not in self.node_renderer.data_source.column_names:
        missing.append("Column 'index' is missing in GraphSource.node_renderer.data_source")
    if 'start' not in self.edge_renderer.data_source.column_names:
        missing.append("Column 'start' is missing in GraphSource.edge_renderer.data_source")
    if 'end' not in self.edge_renderer.data_source.column_names:
        missing.append("Column 'end' is missing in GraphSource.edge_renderer.data_source")
    if missing:
        return ' ,'.join(missing) + ' [%s]' % self