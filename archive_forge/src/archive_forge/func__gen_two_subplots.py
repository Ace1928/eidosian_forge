from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _gen_two_subplots(f, fig, **kwargs):
    """
    Create plot on two subplots forcefully created.
    """
    if 'ax' not in kwargs:
        fig.add_subplot(211)
    yield f(**kwargs)
    if f is pd.plotting.bootstrap_plot:
        assert 'ax' not in kwargs
    else:
        kwargs['ax'] = fig.add_subplot(212)
    yield f(**kwargs)