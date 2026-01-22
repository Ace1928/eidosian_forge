from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _check_legend_labels(axes, labels=None, visible=True):
    """
    Check each axes has expected legend labels

    Parameters
    ----------
    axes : matplotlib Axes object, or its list-like
    labels : list-like
        expected legend labels
    visible : bool
        expected legend visibility. labels are checked only when visible is
        True
    """
    if visible and labels is None:
        raise ValueError('labels must be specified when visible is True')
    axes = _flatten_visible(axes)
    for ax in axes:
        if visible:
            assert ax.get_legend() is not None
            _check_text_labels(ax.get_legend().get_texts(), labels)
        else:
            assert ax.get_legend() is None