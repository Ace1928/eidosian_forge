from collections.abc import Generator
import contextlib
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import accessor
def test_raises_attribute_error():
    with ensure_removed(pd.Series, 'bad'):

        @pd.api.extensions.register_series_accessor('bad')
        class Bad:

            def __init__(self, data) -> None:
                raise AttributeError('whoops')
        with pytest.raises(AttributeError, match='whoops'):
            pd.Series([], dtype=object).bad