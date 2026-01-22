from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def _make_progress(self):
    import holoviews as hv
    import holoviews.plotting.bokeh
    if self._graph:
        root = get_root(self._graph)
        depth = get_depth(root, self._graph)
        breadths = get_breadths(root, self._graph)
        max_breadth = max((len(v) for v in breadths.values()))
    else:
        root = None
        max_breadth, depth = (0, 0)
        breadths = {}
    height = 80 + (max_breadth - 1) * 20
    edges = []
    for src, tgts in self._graph.items():
        for t in tgts:
            edges.append((src, t))
    nodes = []
    for depth, subnodes in breadths.items():
        breadth = len(subnodes)
        step = 1.0 / breadth
        for i, n in enumerate(subnodes[::-1]):
            if n == self._stage:
                state = 'active'
            elif n == self._error:
                state = 'error'
            elif n == self._next_stage:
                state = 'next'
            else:
                state = 'inactive'
            nodes.append((depth, step / 2.0 + i * step, n, state))
    cmap = {'inactive': 'white', 'active': '#5cb85c', 'error': 'red', 'next': 'yellow'}

    def tap_renderer(plot, element):
        from bokeh.models import TapTool
        gr = plot.handles['glyph_renderer']
        tap = plot.state.select_one(TapTool)
        tap.renderers = [gr]
    nodes = hv.Nodes(nodes, ['x', 'y', 'Stage'], 'State').opts(alpha=0, default_tools=['tap'], hooks=[tap_renderer], hover_alpha=0, selection_alpha=0, nonselection_alpha=0, axiswise=True, size=10, backend='bokeh')
    self._progress_sel.source = nodes
    graph = hv.Graph((edges, nodes)).opts(edge_hover_line_color='black', node_color='State', cmap=cmap, tools=[], default_tools=['hover'], selection_policy=None, node_hover_fill_color='gray', axiswise=True, backend='bokeh')
    labels = hv.Labels(nodes, ['x', 'y'], 'Stage').opts(yoffset=-0.3, default_tools=[], axiswise=True, backend='bokeh')
    plot = graph * labels * nodes if self._linear else graph * nodes
    plot.opts(xaxis=None, yaxis=None, min_width=400, responsive=True, show_frame=False, height=height, xlim=(-0.25, depth + 0.25), ylim=(0, 1), default_tools=['hover'], toolbar=None, backend='bokeh')
    return plot