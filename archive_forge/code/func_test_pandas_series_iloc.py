from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@dec.skip_without('pandas')
def test_pandas_series_iloc():
    import pandas as pd
    series = pd.Series([1], index=['a'])
    context = limited(data=series)
    assert guarded_eval('data.iloc[0]', context) == 1