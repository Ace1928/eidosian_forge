import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_order_by():
    table = pa.table({'a': [1, 2, 3, 4], 'b': [1, 3, None, 2]})
    table_source = Declaration('table_source', TableSourceNodeOptions(table))
    ord_opts = OrderByNodeOptions([('b', 'ascending')])
    decl = Declaration.from_sequence([table_source, Declaration('order_by', ord_opts)])
    result = decl.to_table()
    expected = pa.table({'a': [1, 4, 2, 3], 'b': [1, 2, 3, None]})
    assert result.equals(expected)
    ord_opts = OrderByNodeOptions([(field('b'), 'descending')])
    decl = Declaration.from_sequence([table_source, Declaration('order_by', ord_opts)])
    result = decl.to_table()
    expected = pa.table({'a': [2, 4, 1, 3], 'b': [3, 2, 1, None]})
    assert result.equals(expected)
    ord_opts = OrderByNodeOptions([(1, 'descending')], null_placement='at_start')
    decl = Declaration.from_sequence([table_source, Declaration('order_by', ord_opts)])
    result = decl.to_table()
    expected = pa.table({'a': [3, 2, 4, 1], 'b': [None, 3, 2, 1]})
    assert result.equals(expected)
    ord_opts = OrderByNodeOptions([])
    decl = Declaration.from_sequence([table_source, Declaration('order_by', ord_opts)])
    with pytest.raises(ValueError, match='`ordering` must be an explicit non-empty ordering'):
        _ = decl.to_table()
    with pytest.raises(ValueError, match='"decreasing" is not a valid sort order'):
        _ = OrderByNodeOptions([('b', 'decreasing')])
    with pytest.raises(ValueError, match='"start" is not a valid null placement'):
        _ = OrderByNodeOptions([('b', 'ascending')], null_placement='start')