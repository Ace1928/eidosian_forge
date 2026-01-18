import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_max_seq_items_equal_to_n(self, idx):
    with pd.option_context('display.max_seq_items', 6):
        result = idx.__repr__()
        expected = "MultiIndex([('foo', 'one'),\n            ('foo', 'two'),\n            ('bar', 'one'),\n            ('baz', 'two'),\n            ('qux', 'one'),\n            ('qux', 'two')],\n           names=['first', 'second'])"
        assert result == expected