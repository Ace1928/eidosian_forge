import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_lvl_preserves_names_when_target_is_list_or_array():
    idx = MultiIndex.from_product([[0, 1], ['a', 'b']], names=['foo', 'bar'])
    assert idx.reindex([], level=0)[0].names == ['foo', 'bar']
    assert idx.reindex([], level=1)[0].names == ['foo', 'bar']