import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import _check_visible
def _assert_xtickslabels_visibility(self, axes, expected):
    for ax, exp in zip(axes, expected):
        _check_visible(ax.get_xticklabels(), visible=exp)