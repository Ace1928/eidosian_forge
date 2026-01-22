from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _check_box_return_type(returned, return_type, expected_keys=None, check_ax_title=True):
    """
    Check box returned type is correct

    Parameters
    ----------
    returned : object to be tested, returned from boxplot
    return_type : str
        return_type passed to boxplot
    expected_keys : list-like, optional
        group labels in subplot case. If not passed,
        the function checks assuming boxplot uses single ax
    check_ax_title : bool
        Whether to check the ax.title is the same as expected_key
        Intended to be checked by calling from ``boxplot``.
        Normal ``plot`` doesn't attach ``ax.title``, it must be disabled.
    """
    from matplotlib.axes import Axes
    types = {'dict': dict, 'axes': Axes, 'both': tuple}
    if expected_keys is None:
        if return_type is None:
            return_type = 'dict'
        assert isinstance(returned, types[return_type])
        if return_type == 'both':
            assert isinstance(returned.ax, Axes)
            assert isinstance(returned.lines, dict)
    else:
        if return_type is None:
            for r in _flatten_visible(returned):
                assert isinstance(r, Axes)
            return
        assert isinstance(returned, Series)
        assert sorted(returned.keys()) == sorted(expected_keys)
        for key, value in returned.items():
            assert isinstance(value, types[return_type])
            if return_type == 'axes':
                if check_ax_title:
                    assert value.get_title() == key
            elif return_type == 'both':
                if check_ax_title:
                    assert value.ax.get_title() == key
                assert isinstance(value.ax, Axes)
                assert isinstance(value.lines, dict)
            elif return_type == 'dict':
                line = value['medians'][0]
                axes = line.axes
                if check_ax_title:
                    assert axes.get_title() == key
            else:
                raise AssertionError