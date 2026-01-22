import numpy as np
from pathlib import Path
from bokeh.models import CustomJS, CustomAction, PolyEditTool
from holoviews.core.ndmapping import UniformNdMapping
from holoviews.plotting.bokeh.callbacks import (
from holoviews.streams import (
from ...element.geo import _Element, Shape
from ...util import project_extents
from ...models import PolyVertexDrawTool, PolyVertexEditTool
from ...operation import project
from ...streams import PolyVertexEdit, PolyVertexDraw
from .plot import GeoOverlayPlot
class PolyVertexDrawCallback(GeoPolyDrawCallback):

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        element = self.plot.current_frame
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.show_vertices:
            vertex_style = dict({'size': 10}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            kwargs['vertex_renderer'] = r1
        renderer = plot.handles['glyph_renderer']
        tooltip = '%s Draw Tool' % type(element).__name__
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        poly_tool = PolyVertexDrawTool(drag=all((s.drag for s in self.streams)), renderers=[renderer], node_style=stream.node_style, end_style=stream.feature_style, description=tooltip, **kwargs)
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(renderer.data_source.data)
        CDSCallback.initialize(self, plot_id)