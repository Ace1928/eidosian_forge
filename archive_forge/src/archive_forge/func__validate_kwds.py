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
def _validate_kwds(self, kwds):
    kind_opts = self._kind_options.get(self.kind, [])
    kind = self.kind
    eltype = self._kind_mapping[kind]
    if eltype in Store.registry[self._backend_compat]:
        valid_opts = Store.registry[self._backend_compat][eltype].style_opts
    else:
        valid_opts = []
    ds_opts = ['max_px', 'threshold']
    mismatches = sorted((k for k in kwds if k not in kind_opts + ds_opts + valid_opts))
    if not mismatches:
        return
    if 'ax' in mismatches:
        mismatches.pop(mismatches.index('ax'))
        param.main.param.warning('hvPlot does not have the concept of axes, and the ax keyword will be ignored. Compose plots with the * operator to overlay plots or the + operator to lay out plots beside each other instead.')
    if 'figsize' in mismatches:
        mismatches.pop(mismatches.index('figsize'))
        param.main.param.warning('hvPlot does not have the concept of a figure, and the figsize keyword will be ignored. The size of each subplot in a layout is set individually using the width and height options.')
    combined_opts = self._data_options + self._axis_options + self._op_options + self._geo_options + kind_opts + valid_opts
    if self._backend_compat == 'bokeh':
        combined_opts = combined_opts + self._style_options
    for mismatch in mismatches:
        suggestions = difflib.get_close_matches(mismatch, combined_opts)
        param.main.param.warning(f'{mismatch} option not found for {self.kind} plot with {self._backend_compat}; similar options include: {suggestions}')