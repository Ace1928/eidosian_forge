import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_slice_integer(self):
    for index, oob in [(Index(np.arange(5, dtype=np.int64)), False), (RangeIndex(5), False), (Index(np.arange(5, dtype=np.int64) + 10), True)]:
        s = Series(range(5), index=index)
        for idx in [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)]:
            result = s.loc[idx]
            if oob:
                indexer = slice(0, 0)
            else:
                indexer = slice(3, 5)
            self.check(result, s, indexer, False)
        for idx in [slice(-6, 6), slice(-6.0, 6.0)]:
            result = s.loc[idx]
            if oob:
                indexer = slice(0, 0)
            else:
                indexer = slice(-6, 6)
            self.check(result, s, indexer, False)
        msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[-6\\.0\\] of type float'
        with pytest.raises(TypeError, match=msg):
            s[slice(-6.0, 6.0)]
        for idx, res1 in [(slice(2.5, 4), slice(3, 5)), (slice(2, 3.5), slice(2, 4)), (slice(2.5, 3.5), slice(3, 4))]:
            result = s.loc[idx]
            if oob:
                res = slice(0, 0)
            else:
                res = res1
            self.check(result, s, res, False)
            msg = f'cannot do slice indexing on {type(index).__name__} with these indexers \\[(2|3)\\.5\\] of type float'
            with pytest.raises(TypeError, match=msg):
                s[idx]