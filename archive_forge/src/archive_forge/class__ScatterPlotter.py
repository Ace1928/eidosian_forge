from functools import partial
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from ._base import (
from .utils import (
from ._compat import groupby_apply_include_groups
from ._statistics import EstimateAggregator, WeightedAggregator
from .axisgrid import FacetGrid, _facet_docs
from ._docstrings import DocstringComponents, _core_docs
class _ScatterPlotter(_RelationalPlotter):
    _legend_attributes = ['color', 's', 'marker']

    def __init__(self, *, data=None, variables={}, legend=None):
        self._default_size_range = np.r_[0.5, 2] * np.square(mpl.rcParams['lines.markersize'])
        super().__init__(data=data, variables=variables)
        self.legend = legend

    def plot(self, ax, kws):
        data = self.comp_data.dropna()
        if data.empty:
            return
        kws = normalize_kwargs(kws, mpl.collections.PathCollection)
        empty = np.full(len(data), np.nan)
        x = data.get('x', empty)
        y = data.get('y', empty)
        _, inv_x = _get_transform_functions(ax, 'x')
        _, inv_y = _get_transform_functions(ax, 'y')
        x, y = (inv_x(x), inv_y(y))
        if 'style' in self.variables:
            example_level = self._style_map.levels[0]
            example_marker = self._style_map(example_level, 'marker')
            kws.setdefault('marker', example_marker)
        m = kws.get('marker', mpl.rcParams.get('marker', 'o'))
        if not isinstance(m, mpl.markers.MarkerStyle):
            m = mpl.markers.MarkerStyle(m)
        if m.is_filled():
            kws.setdefault('edgecolor', 'w')
        points = ax.scatter(x=x, y=y, **kws)
        if 'hue' in self.variables:
            points.set_facecolors(self._hue_map(data['hue']))
        if 'size' in self.variables:
            points.set_sizes(self._size_map(data['size']))
        if 'style' in self.variables:
            p = [self._style_map(val, 'path') for val in data['style']]
            points.set_paths(p)
        if 'linewidth' not in kws:
            sizes = points.get_sizes()
            linewidth = 0.08 * np.sqrt(np.percentile(sizes, 10))
            points.set_linewidths(linewidth)
            kws['linewidth'] = linewidth
        self._add_axis_labels(ax)
        if self.legend:
            attrs = {'hue': 'color', 'size': 's', 'style': None}
            self.add_legend_data(ax, _scatter_legend_artist, kws, attrs)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(title=self.legend_title)
                adjust_legend_subtitles(legend)