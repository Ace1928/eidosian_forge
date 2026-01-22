from __future__ import annotations
from typing import (
import warnings
from matplotlib.artist import setp
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
import pandas as pd
import pandas.core.common as com
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None:
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        assert remainder == 0, remainder
        labels *= i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)