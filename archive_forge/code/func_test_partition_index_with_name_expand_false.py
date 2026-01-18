from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_partition_index_with_name_expand_false():
    idx = Index(['a,b', 'c,d'], name='xxx')
    result = idx.str.partition(',', expand=False)
    expected = Index(np.array([('a', ',', 'b'), ('c', ',', 'd')]), name='xxx')
    assert result.nlevels == 1
    tm.assert_index_equal(result, expected)