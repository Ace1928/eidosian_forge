import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_table_join_keys_order():
    t1 = pa.table({'colB': [10, 20, 60], 'colA': [1, 2, 6], 'colVals': ['a', 'b', 'f']})
    t2 = pa.table({'colVals': ['Z', 'B', 'A'], 'colX': [99, 2, 1]})
    result = _perform_join('full outer', t1, 'colA', t2, 'colX', left_suffix='_l', right_suffix='_r', coalesce_keys=True)
    result = result.combine_chunks()
    result = result.sort_by('colA')
    assert result == pa.table({'colB': [10, 20, 60, None], 'colA': [1, 2, 6, 99], 'colVals_l': ['a', 'b', 'f', None], 'colVals_r': ['A', 'B', None, 'Z']})