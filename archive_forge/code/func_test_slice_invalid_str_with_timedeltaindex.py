from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_slice_invalid_str_with_timedeltaindex(self, tdi, frame_or_series, indexer_sl):
    obj = frame_or_series(range(10), index=tdi)
    msg = 'cannot do slice indexing on TimedeltaIndex with these indexers \\[foo\\] of type str'
    with pytest.raises(TypeError, match=msg):
        indexer_sl(obj)['foo':]
    with pytest.raises(TypeError, match=msg):
        indexer_sl(obj)['foo':-1]
    with pytest.raises(TypeError, match=msg):
        indexer_sl(obj)[:'foo']
    with pytest.raises(TypeError, match=msg):
        indexer_sl(obj)[tdi[0]:'foo']