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
class GeoSelectionXYCallback(SelectionXYCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        if skip(self, msg, ('x_selection', 'y_selection')) or not all((isinstance(sel, tuple) for sel in msg.values())):
            return msg
        plot = get_cb_plot(self)
        x0, x1 = msg['x_selection']
        y0, y1 = msg['y_selection']
        l, b, r, t = bounds = project_extents((x0, y0, x1, y1), plot.projection, plot.current_frame.crs)
        return {'x_selection': (l, r), 'y_selection': (b, t), 'bounds': bounds}