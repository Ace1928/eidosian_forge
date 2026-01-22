import holoviews as _hv
import numpy as np
import panel as pn
import param
from holoviews.core.util import datetime_types, dt_to_int, is_number, max_range
from holoviews.element import tile_sources
from holoviews.plotting.util import list_cmaps
from panel.viewable import Viewer
from .converter import HoloViewsConverter as _hvConverter
from .plotting import hvPlot as _hvPlot
from .util import is_geodataframe, is_xarray, instantiate_crs_str
def _toggle_controls(self, event=None):
    visible = True
    if event and event.new in ('table', 'dataset'):
        parameters = ['kind', 'columns']
        visible = False
    elif event and event.new in KINDS['2d']:
        parameters = ['kind', 'x', 'y', 'by', 'groupby']
    elif event and event.new in ('hist', 'kde', 'density'):
        self.x = None
        parameters = ['kind', 'y_multi', 'by', 'groupby']
    else:
        parameters = ['kind', 'x', 'y_multi', 'by', 'groupby']
    self._controls.parameters = parameters
    tabs = [('Fields', self._controls)]
    if visible:
        tabs += [('Axes', self.axes), ('Labels', self.labels), ('Style', self.style), ('Operations', self.operations), ('Geographic', self.geographic), ('Advanced', self.advanced)]
        if event and event.new not in ('area', 'kde', 'line', 'ohlc', 'rgb', 'step'):
            tabs.insert(5, ('Colormapping', self.colormapping))
    self._control_tabs[:] = tabs