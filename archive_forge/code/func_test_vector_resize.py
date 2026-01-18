from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
@pytest.mark.parametrize('nvals', [0, 10])
@pytest.mark.parametrize('htable, uniques, dtype, safely_resizes', [(ht.PyObjectHashTable, ht.ObjectVector, 'object', False), (ht.StringHashTable, ht.ObjectVector, 'object', True), (ht.Float64HashTable, ht.Float64Vector, 'float64', False), (ht.Int64HashTable, ht.Int64Vector, 'int64', False), (ht.Int32HashTable, ht.Int32Vector, 'int32', False), (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False)])
def test_vector_resize(self, writable, htable, uniques, dtype, safely_resizes, nvals):
    vals = np.array(range(1000), dtype=dtype)
    vals.setflags(write=writable)
    htable = htable()
    uniques = uniques()
    htable.get_labels(vals[:nvals], uniques, 0, -1)
    tmp = uniques.to_array()
    oldshape = tmp.shape
    if safely_resizes:
        htable.get_labels(vals, uniques, 0, -1)
    else:
        with pytest.raises(ValueError, match='external reference.*'):
            htable.get_labels(vals, uniques, 0, -1)
    uniques.to_array()
    assert tmp.shape == oldshape