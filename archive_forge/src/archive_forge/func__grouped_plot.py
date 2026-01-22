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
def _grouped_plot(plotf, data: Series | DataFrame, column=None, by=None, numeric_only: bool=True, figsize: tuple[float, float] | None=None, sharex: bool=True, sharey: bool=True, layout=None, rot: float=0, ax=None, **kwargs):
    if figsize == 'default':
        raise ValueError("figsize='default' is no longer supported. Specify figure size by tuple instead")
    grouped = data.groupby(by)
    if column is not None:
        grouped = grouped[column]
    naxes = len(grouped)
    fig, axes = create_subplots(naxes=naxes, figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout)
    _axes = flatten_axes(axes)
    for i, (key, group) in enumerate(grouped):
        ax = _axes[i]
        if numeric_only and isinstance(group, ABCDataFrame):
            group = group._get_numeric_data()
        plotf(group, ax, **kwargs)
        ax.set_title(pprint_thing(key))
    return (fig, axes)