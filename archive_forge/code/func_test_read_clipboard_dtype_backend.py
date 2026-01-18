from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('engine', ['c', 'python'])
def test_read_clipboard_dtype_backend(self, clipboard, string_storage, dtype_backend, engine):
    if string_storage == 'pyarrow' or dtype_backend == 'pyarrow':
        pa = pytest.importorskip('pyarrow')
    if string_storage == 'python':
        string_array = StringArray(np.array(['x', 'y'], dtype=np.object_))
        string_array_na = StringArray(np.array(['x', NA], dtype=np.object_))
    elif dtype_backend == 'pyarrow' and engine != 'c':
        pa = pytest.importorskip('pyarrow')
        from pandas.arrays import ArrowExtensionArray
        string_array = ArrowExtensionArray(pa.array(['x', 'y']))
        string_array_na = ArrowExtensionArray(pa.array(['x', None]))
    else:
        string_array = ArrowStringArray(pa.array(['x', 'y']))
        string_array_na = ArrowStringArray(pa.array(['x', None]))
    text = 'a,b,c,d,e,f,g,h,i\nx,1,4.0,x,2,4.0,,True,False\ny,2,5.0,,,,,False,'
    clipboard.setText(text)
    with pd.option_context('mode.string_storage', string_storage):
        result = read_clipboard(sep=',', dtype_backend=dtype_backend, engine=engine)
    expected = DataFrame({'a': string_array, 'b': Series([1, 2], dtype='Int64'), 'c': Series([4.0, 5.0], dtype='Float64'), 'd': string_array_na, 'e': Series([2, NA], dtype='Int64'), 'f': Series([4.0, NA], dtype='Float64'), 'g': Series([NA, NA], dtype='Int64'), 'h': Series([True, False], dtype='boolean'), 'i': Series([False, NA], dtype='boolean')})
    if dtype_backend == 'pyarrow':
        from pandas.arrays import ArrowExtensionArray
        expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
        expected['g'] = ArrowExtensionArray(pa.array([None, None]))
    tm.assert_frame_equal(result, expected)