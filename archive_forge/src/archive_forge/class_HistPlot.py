from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
class HistPlot(LinePlot):

    @property
    def _kind(self) -> Literal['hist', 'kde']:
        return 'hist'

    def __init__(self, data, bins: int | np.ndarray | list[np.ndarray]=10, bottom: int | np.ndarray=0, *, range=None, weights=None, **kwargs) -> None:
        if is_list_like(bottom):
            bottom = np.array(bottom)
        self.bottom = bottom
        self._bin_range = range
        self.weights = weights
        self.xlabel = kwargs.get('xlabel')
        self.ylabel = kwargs.get('ylabel')
        MPLPlot.__init__(self, data, **kwargs)
        self.bins = self._adjust_bins(bins)

    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
        if is_integer(bins):
            if self.by is not None:
                by_modified = unpack_single_str_list(self.by)
                grouped = self.data.groupby(by_modified)[self.columns]
                bins = [self._calculate_bins(group, bins) for key, group in grouped]
            else:
                bins = self._calculate_bins(self.data, bins)
        return bins

    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        """Calculate bins given data"""
        nd_values = data.infer_objects(copy=False)._get_numeric_data()
        values = np.ravel(nd_values)
        values = values[~isna(values)]
        hist, bins = np.histogram(values, bins=bins, range=self._bin_range)
        return bins

    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style=None, bottom: int | np.ndarray=0, column_num: int=0, stacking_id=None, *, bins, **kwds):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)
        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds['label'])
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _make_plot(self, fig: Figure) -> None:
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()
        data = create_iter_data_given_by(self.data, self._kind) if self.by is not None else self.data
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            ax = self._get_ax(i)
            kwds = self.kwds.copy()
            if self.color is not None:
                kwds['color'] = self.color
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds['label'] = label
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds['style'] = style
            self._make_plot_keywords(kwds, y)
            if self.by is not None:
                kwds['bins'] = kwds['bins'][i]
                kwds['label'] = self.columns
                kwds.pop('color')
            if self.weights is not None:
                kwds['weights'] = type(self)._get_column_weights(self.weights, i, y)
            y = reformat_hist_y_given_by(y, self.by)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            if self.by is not None:
                ax.set_title(pprint_thing(label))
            self._append_legend_handles_labels(artists[0], label)

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """merge BoxPlot/KdePlot properties to passed kwds"""
        kwds['bottom'] = self.bottom
        kwds['bins'] = self.bins

    @final
    @staticmethod
    def _get_column_weights(weights, i: int, y):
        if weights is not None:
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError('weights must have the same shape as data, or be a single column') from err
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.orientation == 'horizontal':
            ax.set_xlabel('Frequency' if self.xlabel is None else self.xlabel)
            ax.set_ylabel(self.ylabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel('Frequency' if self.ylabel is None else self.ylabel)

    @property
    def orientation(self) -> PlottingOrientation:
        if self.kwds.get('orientation', None) == 'horizontal':
            return 'horizontal'
        else:
            return 'vertical'