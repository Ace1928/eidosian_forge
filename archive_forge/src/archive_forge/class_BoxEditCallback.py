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
class BoxEditCallback(GlyphDrawCallback):
    attributes = {'data': 'cds.data'}
    models = ['cds']

    def _path_initialize(self):
        plot = self.plot
        cds = plot.handles['cds']
        data = cds.data
        element = self.plot.current_frame
        l, b, r, t = ([], [], [], [])
        for x, y in zip(data['xs'], data['ys']):
            x0, x1 = (np.nanmin(x), np.nanmax(x))
            y0, y1 = (np.nanmin(y), np.nanmax(y))
            l.append(x0)
            b.append(y0)
            r.append(x1)
            t.append(y1)
        data = {'left': l, 'bottom': b, 'right': r, 'top': t}
        data.update({vd.name: element.dimension_values(vd, expanded=False) for vd in element.vdims})
        cds.data.update(data)
        style = self.plot.style[self.plot.cyclic_index]
        style.pop('cmap', None)
        r1 = plot.state.quad(left='left', bottom='bottom', right='right', top='top', source=cds, **style)
        if plot.handles['glyph_renderer'] in self.plot.state.renderers:
            self.plot.state.renderers.remove(plot.handles['glyph_renderer'])
        data = self._process_msg({'data': data})['data']
        for stream in self.streams:
            stream.update(data=data)
        return r1

    def initialize(self, plot_id=None):
        from .path import PathPlot
        stream = self.streams[0]
        cds = self.plot.handles['cds']
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        renderer = self.plot.handles['glyph_renderer']
        if isinstance(self.plot, PathPlot):
            renderer = self._path_initialize()
        if stream.styles:
            self._create_style_callback(cds, renderer.glyph)
        if bokeh33:
            box_tool = BoxEditTool(renderers=[renderer], **kwargs)
            self.plot.state.tools.append(box_tool)
        else:
            warn('BoxEditTool requires Bokeh >= 3.3')
        self._update_cds_vdims(cds.data)
        super(CDSCallback, self).initialize()

    def _process_msg(self, msg):
        data = super()._process_msg(msg)
        if 'data' not in data:
            return {}
        values = dict(data['data'])
        values['x0'] = values.pop('left')
        values['y0'] = values.pop('bottom')
        values['x1'] = values.pop('right')
        values['y1'] = values.pop('top')
        msg = {'data': values}
        self._update_cds_vdims(msg['data'])
        return self._transform(msg)