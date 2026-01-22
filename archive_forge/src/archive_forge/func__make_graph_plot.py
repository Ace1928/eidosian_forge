from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def _make_graph_plot(self) -> Plot:
    """ Builds the graph portion of the final model.

        """
    import networkx as nx
    nodes = nx.nx_agraph.graphviz_layout(self._graph, prog='dot')
    node_x, node_y = zip(*nodes.values())
    models = [self._graph.nodes[x]['model'] for x in nodes]
    node_id = list(nodes.keys())
    node_source = ColumnDataSource({'x': node_x, 'y': node_y, 'index': node_id, 'model': models})
    edge_x_coords = []
    edge_y_coords = []
    for start_node, end_node in self._graph.edges:
        edge_x_coords.extend([[nodes[start_node][0], nodes[end_node][0]]])
        edge_y_coords.extend([[nodes[start_node][1], nodes[end_node][1]]])
    edge_source = ColumnDataSource({'xs': edge_x_coords, 'ys': edge_y_coords})
    p2 = Plot(outline_line_alpha=0.0)
    xinterval = max(max(node_x) - min(node_x), 200)
    yinterval = max(max(node_y) - min(node_y), 200)
    p2.x_range = Range1d(start=min(node_x) - 0.15 * xinterval, end=max(node_x) + 0.15 * xinterval)
    p2.y_range = Range1d(start=min(node_y) - 0.15 * yinterval, end=max(node_y) + 0.15 * yinterval)
    node_renderer = GlyphRenderer(data_source=node_source, glyph=Scatter(x='x', y='y', size=15, fill_color='lightblue'), nonselection_glyph=Scatter(x='x', y='y', size=15, fill_color='lightblue'), selection_glyph=Scatter(x='x', y='y', size=15, fill_color='green'))
    edge_renderer = GlyphRenderer(data_source=edge_source, glyph=MultiLine(xs='xs', ys='ys'))
    node_hover_tool = HoverTool(tooltips=[('id', '@index'), ('model', '@model')])
    node_hover_tool.renderers = [node_renderer]
    tap_tool = TapTool()
    tap_tool.renderers = [node_renderer]
    labels = LabelSet(x='x', y='y', text='model', source=node_source, text_font_size='8pt', x_offset=-20, y_offset=7)
    help = Label(x=20, y=20, x_units='screen', y_units='screen', text_font_size='8pt', text_font_style='italic', text='Click on a model to see its attributes')
    p2.add_layout(help)
    p2.add_layout(edge_renderer)
    p2.add_layout(node_renderer)
    p2.tools.extend([node_hover_tool, tap_tool, BoxZoomTool(), ResetTool(), PanTool()])
    p2.renderers.append(labels)
    self._node_source = node_source
    self._edge_source = edge_source
    return p2