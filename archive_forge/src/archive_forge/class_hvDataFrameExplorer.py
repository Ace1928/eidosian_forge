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
class hvDataFrameExplorer(hvPlotExplorer):
    z = param.Selector()
    kind = param.Selector(default='scatter', objects=KINDS['all'])

    @property
    def _var_name(self):
        return 'df'

    @property
    def xcat(self):
        if self.kind in ('bar', 'box', 'violin'):
            return False
        values = self._data[self.x]
        return values.dtype.kind in 'OSU'

    @property
    def _x(self):
        return self._converter.x or self._converter.variables[0] if self.x is None else self.x

    @property
    def _y(self):
        if 'y_multi' in self._controls.parameters and self.y_multi:
            y = self.y_multi
        elif 'y_multi' not in self._controls.parameters and self.y:
            y = self.y
        else:
            y = self._converter._process_chart_y(self._data, self._x, None, self._single_y)
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        return y

    @property
    def _groups(self):
        return ['dataframe']

    @param.depends('x')
    def xlim(self):
        if self._x == 'index':
            values = self._data.index.values
        else:
            try:
                values = self._data[self._x]
            except:
                return (0, 1)
        if hasattr(values, 'compute_chunk_sizes'):
            values = values.compute_chunk_sizes()
        if values.dtype.kind in 'OSU':
            return None
        elif not len(values):
            return (np.nan, np.nan)
        return (np.nanmin(values), np.nanmax(values))

    @param.depends('y', 'y_multi')
    def ylim(self):
        y = self._y
        if not isinstance(y, list):
            y = [y]
        values = [ys for ys in (self._data[y] for y in y) if len(ys)]
        if not len(values):
            return (np.nan, np.nan)
        return max_range([(np.nanmin(vs), np.nanmax(vs)) for vs in values])