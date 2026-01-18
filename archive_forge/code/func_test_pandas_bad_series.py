from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@dec.skip_without('pandas')
def test_pandas_bad_series():
    import pandas as pd

    class BadItemSeries(pd.Series):

        def __getitem__(self, key):
            return 'CUSTOM_ITEM'

    class BadAttrSeries(pd.Series):

        def __getattr__(self, key):
            return 'CUSTOM_ATTR'
    bad_series = BadItemSeries([1], index=['a'])
    context = limited(data=bad_series)
    with pytest.raises(GuardRejection):
        guarded_eval('data["a"]', context)
    with pytest.raises(GuardRejection):
        guarded_eval('data["c"]', context)
    assert guarded_eval('data.a', context) == 'CUSTOM_ITEM'
    context = unsafe(data=bad_series)
    assert guarded_eval('data["a"]', context) == 'CUSTOM_ITEM'
    bad_attr_series = BadAttrSeries([1], index=['a'])
    context = limited(data=bad_attr_series)
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(GuardRejection):
        guarded_eval('data.a', context)