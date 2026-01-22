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
def _create_vertex_split_link(self, action, poly_renderer, vertex_renderer, vertex_tool):
    cb = CustomJS(code=self.split_code, args={'poly': poly_renderer, 'vertex': vertex_renderer, 'tool': vertex_tool})
    action.callback = cb