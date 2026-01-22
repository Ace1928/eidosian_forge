import asyncio
import base64
import time
from collections import defaultdict
import numpy as np
from bokeh.models import (
from panel.io.state import set_curdoc, state
from ...core.options import CallbackError
from ...core.util import datetime_types, dimension_sanitizer, dt64_to_dt, isequal
from ...element import Table
from ...streams import (
from ...util.warnings import warn
from .util import bokeh33, convert_timestamp
class PolyDrawCallback(GlyphDrawCallback):

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        cds = self.plot.handles['cds']
        glyph = self.plot.handles['glyph']
        renderers = [plot.handles['glyph_renderer']]
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.show_vertices:
            vertex_style = dict({'size': 10}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            kwargs['vertex_renderer'] = r1
        if stream.styles:
            self._create_style_callback(cds, glyph)
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        poly_tool = PolyDrawTool(drag=all((s.drag for s in self.streams)), renderers=renderers, **kwargs)
        plot.state.tools.append(poly_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)

    def _process_msg(self, msg):
        self._update_cds_vdims(msg['data'])
        return super()._process_msg(msg)

    def _update_cds_vdims(self, data):
        """
        Add any value dimensions not already in the data ensuring the
        element can be reconstituted in entirety.
        """
        element = self.plot.current_frame
        stream = self.streams[0]
        interface = element.interface
        scalar_kwargs = {'per_geom': True} if interface.multi else {}
        for d in element.vdims:
            scalar = element.interface.isunique(element, d, **scalar_kwargs)
            dim = dimension_sanitizer(d.name)
            if dim not in data:
                if scalar:
                    values = element.dimension_values(d, not scalar)
                else:
                    values = [arr[:, 0] for arr in element.split(datatype='array', dimensions=[dim])]
                if len(values) != len(data['xs']):
                    values = np.concatenate([values, [stream.empty_value]])
                data[dim] = values