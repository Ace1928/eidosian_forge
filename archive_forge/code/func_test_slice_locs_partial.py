from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_locs_partial(self, idx):
    sorted_idx, _ = idx.sortlevel(0)
    result = sorted_idx.slice_locs(('foo', 'two'), ('qux', 'one'))
    assert result == (1, 5)
    result = sorted_idx.slice_locs(None, ('qux', 'one'))
    assert result == (0, 5)
    result = sorted_idx.slice_locs(('foo', 'two'), None)
    assert result == (1, len(sorted_idx))
    result = sorted_idx.slice_locs('bar', 'baz')
    assert result == (2, 4)