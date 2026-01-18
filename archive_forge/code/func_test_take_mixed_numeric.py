import pytest
import pandas._testing as tm
def test_take_mixed_numeric(self, mixed_float_frame, mixed_int_frame):
    order = [1, 2, 0, 3]
    for df in [mixed_float_frame, mixed_int_frame]:
        result = df.take(order, axis=0)
        expected = df.reindex(df.index.take(order))
        tm.assert_frame_equal(result, expected)
        result = df.take(order, axis=1)
        expected = df.loc[:, ['B', 'C', 'A', 'D']]
        tm.assert_frame_equal(result, expected)