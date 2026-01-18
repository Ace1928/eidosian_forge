from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_with_name_index():
    idx = Index(['a,b', 'c,d'], name='xxx')
    res = idx.str.split(',')
    exp = Index([['a', 'b'], ['c', 'd']], name='xxx')
    assert res.nlevels == 1
    tm.assert_index_equal(res, exp)
    res = idx.str.split(',', expand=True)
    exp = MultiIndex.from_tuples([('a', 'b'), ('c', 'd')])
    assert res.nlevels == 2
    tm.assert_index_equal(res, exp)