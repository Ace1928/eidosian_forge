import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_axis_copy(self, obj, using_copy_on_write):
    new_index = list('abcd')[:len(obj)]
    orig = obj.iloc[:]
    expected = obj.copy()
    expected.index = new_index
    result = obj.set_axis(new_index, axis=0, copy=True)
    tm.assert_equal(expected, result)
    assert result is not obj
    if not using_copy_on_write:
        if obj.ndim == 1:
            assert not tm.shares_memory(result, obj)
        else:
            assert not any((tm.shares_memory(result.iloc[:, i], obj.iloc[:, i]) for i in range(obj.shape[1])))
    result = obj.set_axis(new_index, axis=0, copy=False)
    tm.assert_equal(expected, result)
    assert result is not obj
    if obj.ndim == 1:
        assert tm.shares_memory(result, obj)
    else:
        assert all((tm.shares_memory(result.iloc[:, i], obj.iloc[:, i]) for i in range(obj.shape[1])))
    result = obj.set_axis(new_index, axis=0)
    tm.assert_equal(expected, result)
    assert result is not obj
    if using_copy_on_write:
        if obj.ndim == 1:
            assert tm.shares_memory(result, obj)
        else:
            assert any((tm.shares_memory(result.iloc[:, i], obj.iloc[:, i]) for i in range(obj.shape[1])))
    elif obj.ndim == 1:
        assert not tm.shares_memory(result, obj)
    else:
        assert not any((tm.shares_memory(result.iloc[:, i], obj.iloc[:, i]) for i in range(obj.shape[1])))
    res = obj.set_axis(new_index, copy=False)
    tm.assert_equal(expected, res)
    if res.ndim == 1:
        assert tm.shares_memory(res, orig)
    else:
        assert all((tm.shares_memory(res.iloc[:, i], orig.iloc[:, i]) for i in range(res.shape[1])))