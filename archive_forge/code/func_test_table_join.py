from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_join():
    t1 = pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f']})
    t2 = pa.table({'colB': [99, 2, 1], 'col3': ['Z', 'B', 'A']})
    result = t1.join(t2, 'colA', 'colB')
    assert result.combine_chunks() == pa.table({'colA': [1, 2, 6], 'col2': ['a', 'b', 'f'], 'col3': ['A', 'B', None]})
    result = t1.join(t2, 'colA', 'colB', join_type='full outer')
    assert result.combine_chunks().sort_by('colA') == pa.table({'colA': [1, 2, 6, 99], 'col2': ['a', 'b', 'f', None], 'col3': ['A', 'B', None, 'Z']})