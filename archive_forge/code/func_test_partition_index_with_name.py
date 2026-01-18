from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_partition_index_with_name():
    idx = Index(['a,b', 'c,d'], name='xxx')
    result = idx.str.partition(',')
    expected = MultiIndex.from_tuples([('a', ',', 'b'), ('c', ',', 'd')])
    assert result.nlevels == 3
    tm.assert_index_equal(result, expected)