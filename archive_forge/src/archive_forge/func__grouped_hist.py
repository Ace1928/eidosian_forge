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
def _grouped_hist(data: Series | DataFrame, column=None, by=None, ax=None, bins: int=50, figsize: tuple[float, float] | None=None, layout=None, sharex: bool=False, sharey: bool=False, rot: float=90, grid: bool=True, xlabelsize: int | None=None, xrot=None, ylabelsize: int | None=None, yrot=None, legend: bool=False, **kwargs):
    """
    Grouped histogram

    Parameters
    ----------
    data : Series/DataFrame
    column : object, optional
    by : object, optional
    ax : axes, optional
    bins : int, default 50
    figsize : tuple, optional
    layout : optional
    sharex : bool, default False
    sharey : bool, default False
    rot : float, default 90
    grid : bool, default True
    legend: : bool, default False
    kwargs : dict, keyword arguments passed to matplotlib.Axes.hist

    Returns
    -------
    collection of Matplotlib Axes
    """
    if legend:
        assert 'label' not in kwargs
        if data.ndim == 1:
            kwargs['label'] = data.name
        elif column is None:
            kwargs['label'] = data.columns
        else:
            kwargs['label'] = column

    def plot_group(group, ax) -> None:
        ax.hist(group.dropna().values, bins=bins, **kwargs)
        if legend:
            ax.legend()
    if xrot is None:
        xrot = rot
    fig, axes = _grouped_plot(plot_group, data, column=column, by=by, sharex=sharex, sharey=sharey, ax=ax, figsize=figsize, layout=layout, rot=rot)
    set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3)
    return axes