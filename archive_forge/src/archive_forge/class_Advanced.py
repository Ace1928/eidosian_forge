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
class Advanced(Controls):
    opts = param.Dict(label='HoloViews .opts()', doc='\n        Options applied via HoloViews .opts().\n        Examples:\n        - image: {"color_levels": 11}\n        - line: {"line_dash": "dashed"}\n        - scatter: {\'size\': 5, \'marker\': \'^\'}')
    _widgets_kwargs = {'opts': {'placeholder': "{'size': 5, 'marker': '^'}"}}