from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_join_unique_key():
    t1 = pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f']})
    t2 = pa.table({'colA': [99, 2, 1], 'col3': ['Z', 'B', 'A']})
    result = t1.join(t2, 'colA')
    assert result.combine_chunks() == pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f'], 'col3': ['A', 'B', None]})
    result = t1.join(t2, 'colA', join_type='full outer', right_suffix='_r')
    assert result.combine_chunks().sort_by('colA') == pa.table({'colA': [1, 2, 6, 99], 'col2': ['a', 'b', 'f', None], 'col3': ['A', 'B', None, 'Z']})