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
def _process_chart_y(self, data, x, y, single_y):
    """This should happen after _process_chart_x"""
    y = y or self.y
    if y is None:
        ys = [c for c in data.columns if c not in [x] + self.by + self.groupby + self.grid]
        if len(ys) > 1:
            from pandas.api.types import is_numeric_dtype as isnum
            num_ys = [dim for dim in ys if isnum(data[dim])]
            if len(num_ys) >= 1:
                ys = num_ys
        y = ys[0] if len(ys) == 1 or single_y else ys
    return y