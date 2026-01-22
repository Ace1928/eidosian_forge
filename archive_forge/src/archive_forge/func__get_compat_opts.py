import difflib
from functools import partial
import param
import holoviews as hv
import pandas as pd
import numpy as np
import colorcet as cc
from bokeh.models import HoverTool
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.core.overlay import NdOverlay
from holoviews.core.options import Store, Cycle, Palette
from holoviews.core.layout import NdLayout
from holoviews.core.util import max_range
from holoviews.element import (
from holoviews.plotting.bokeh import OverlayPlot, colormap_generator
from holoviews.plotting.util import process_cmap
from holoviews.operation import histogram, apply_when
from holoviews.streams import Buffer, Pipe
from holoviews.util.transform import dim
from packaging.version import Version
from pandas import DatetimeIndex, MultiIndex
from .backend_transforms import _transfer_opts_cur_backend
from .util import (
from .utilities import hvplot_extension
def _get_compat_opts(self, el_name, **custom):
    cur_opts = self._get_opts(el_name, backend='bokeh', **custom)
    if self._backend_compat != 'bokeh':
        compat_opts = self._get_opts(el_name, backend=self._backend_compat)
        compat_opts = {k: v for k, v in compat_opts.items() if k not in cur_opts}
    else:
        compat_opts = {}
    return (cur_opts, compat_opts)