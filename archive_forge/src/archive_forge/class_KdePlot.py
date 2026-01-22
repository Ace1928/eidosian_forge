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
class KdePlot(HistPlot):

    @property
    def _kind(self) -> Literal['kde']:
        return 'kde'

    @property
    def orientation(self) -> Literal['vertical']:
        return 'vertical'

    def __init__(self, data, bw_method=None, ind=None, *, weights=None, **kwargs) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        self.bw_method = bw_method
        self.ind = ind
        self.weights = weights

    @staticmethod
    def _get_ind(y: np.ndarray, ind):
        if ind is None:
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(np.nanmin(y) - 0.5 * sample_range, np.nanmax(y) + 0.5 * sample_range, 1000)
        elif is_integer(ind):
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(np.nanmin(y) - 0.5 * sample_range, np.nanmax(y) + 0.5 * sample_range, ind)
        return ind

    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style=None, bw_method=None, ind=None, column_num=None, stacking_id: int | None=None, **kwds):
        from scipy.stats import gaussian_kde
        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method)
        y = gkde.evaluate(ind)
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        kwds['bw_method'] = self.bw_method
        kwds['ind'] = type(self)._get_ind(y, ind=self.ind)

    def _post_plot_logic(self, ax: Axes, data) -> None:
        ax.set_ylabel('Density')