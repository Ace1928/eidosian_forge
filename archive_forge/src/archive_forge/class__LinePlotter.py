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
class _LinePlotter(_RelationalPlotter):
    _legend_attributes = ['color', 'linewidth', 'marker', 'dashes']

    def __init__(self, *, data=None, variables={}, estimator=None, n_boot=None, seed=None, errorbar=None, sort=True, orient='x', err_style=None, err_kws=None, legend=None):
        self._default_size_range = np.r_[0.5, 2] * mpl.rcParams['lines.linewidth']
        super().__init__(data=data, variables=variables)
        self.estimator = estimator
        self.errorbar = errorbar
        self.n_boot = n_boot
        self.seed = seed
        self.sort = sort
        self.orient = orient
        self.err_style = err_style
        self.err_kws = {} if err_kws is None else err_kws
        self.legend = legend

    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""
        kws = normalize_kwargs(kws, mpl.lines.Line2D)
        kws.setdefault('markeredgewidth', 0.75)
        kws.setdefault('markeredgecolor', 'w')
        err_kws = self.err_kws.copy()
        if self.err_style == 'band':
            err_kws.setdefault('alpha', 0.2)
        elif self.err_style == 'bars':
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))
        weighted = 'weight' in self.plot_data
        agg = (WeightedAggregator if weighted else EstimateAggregator)(self.estimator, self.errorbar, n_boot=self.n_boot, seed=self.seed)
        orient = self.orient
        if orient not in {'x', 'y'}:
            err = f"`orient` must be either 'x' or 'y', not {orient!r}."
            raise ValueError(err)
        other = {'x': 'y', 'y': 'x'}[orient]
        grouping_vars = ('hue', 'size', 'style')
        for sub_vars, sub_data in self.iter_data(grouping_vars, from_comp_data=True):
            if self.sort:
                sort_vars = ['units', orient, other]
                sort_cols = [var for var in sort_vars if var in self.variables]
                sub_data = sub_data.sort_values(sort_cols)
            if self.estimator is not None and sub_data[orient].value_counts().max() > 1:
                if 'units' in self.variables:
                    err = 'estimator must be None when specifying units'
                    raise ValueError(err)
                grouped = sub_data.groupby(orient, sort=self.sort)
                sub_data = grouped.apply(agg, other, **groupby_apply_include_groups(False)).reset_index()
            else:
                sub_data[f'{other}min'] = np.nan
                sub_data[f'{other}max'] = np.nan
            for var in 'xy':
                _, inv = _get_transform_functions(ax, var)
                for col in sub_data.filter(regex=f'^{var}'):
                    sub_data[col] = inv(sub_data[col])
            if 'units' in self.variables:
                lines = []
                for _, unit_data in sub_data.groupby('units'):
                    lines.extend(ax.plot(unit_data['x'], unit_data['y'], **kws))
            else:
                lines = ax.plot(sub_data['x'], sub_data['y'], **kws)
            for line in lines:
                if 'hue' in sub_vars:
                    line.set_color(self._hue_map(sub_vars['hue']))
                if 'size' in sub_vars:
                    line.set_linewidth(self._size_map(sub_vars['size']))
                if 'style' in sub_vars:
                    attributes = self._style_map(sub_vars['style'])
                    if 'dashes' in attributes:
                        line.set_dashes(attributes['dashes'])
                    if 'marker' in attributes:
                        line.set_marker(attributes['marker'])
            line_color = line.get_color()
            line_alpha = line.get_alpha()
            line_capstyle = line.get_solid_capstyle()
            if self.estimator is not None and self.errorbar is not None:
                if self.err_style == 'band':
                    func = {'x': ax.fill_between, 'y': ax.fill_betweenx}[orient]
                    func(sub_data[orient], sub_data[f'{other}min'], sub_data[f'{other}max'], color=line_color, **err_kws)
                elif self.err_style == 'bars':
                    error_param = {f'{other}err': (sub_data[other] - sub_data[f'{other}min'], sub_data[f'{other}max'] - sub_data[other])}
                    ebars = ax.errorbar(sub_data['x'], sub_data['y'], **error_param, linestyle='', color=line_color, alpha=line_alpha, **err_kws)
                    for obj in ebars.get_children():
                        if isinstance(obj, mpl.collections.LineCollection):
                            obj.set_capstyle(line_capstyle)
        self._add_axis_labels(ax)
        if self.legend:
            legend_artist = partial(mpl.lines.Line2D, xdata=[], ydata=[])
            attrs = {'hue': 'color', 'size': 'linewidth', 'style': None}
            self.add_legend_data(ax, legend_artist, kws, attrs)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(title=self.legend_title)
                adjust_legend_subtitles(legend)