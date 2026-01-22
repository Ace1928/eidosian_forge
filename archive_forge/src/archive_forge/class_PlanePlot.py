from __future__ import annotations
from abc import (
from collections.abc import (
from typing import (
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib import tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.timeseries import (
from pandas.plotting._matplotlib.tools import (
class PlanePlot(MPLPlot, ABC):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """
    _layout_type = 'single'

    def __init__(self, data, x, y, **kwargs) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + ' requires an x and y column')
        if is_integer(x) and (not self.data.columns._holds_integer()):
            x = self.data.columns[x]
        if is_integer(y) and (not self.data.columns._holds_integer()):
            y = self.data.columns[y]
        self.x = x
        self.y = y

    @final
    def _get_nseries(self, data: Series | DataFrame) -> int:
        return 1

    @final
    def _post_plot_logic(self, ax: Axes, data) -> None:
        x, y = (self.x, self.y)
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds):
        img = ax.collections[-1]
        return fig.colorbar(img, ax=ax, **kwds)