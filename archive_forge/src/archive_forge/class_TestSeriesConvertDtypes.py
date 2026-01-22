from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
class TestSeriesConvertDtypes:

    @pytest.mark.parametrize('params', product(*[(True, False)] * 5))
    def test_convert_dtypes(self, test_cases, params, using_infer_string):
        data, maindtype, expected_default, expected_other = test_cases
        if hasattr(data, 'dtype') and lib.is_np_dtype(data.dtype, 'M') and isinstance(maindtype, pd.DatetimeTZDtype):
            msg = 'Cannot use .astype to convert from timezone-naive dtype'
            with pytest.raises(TypeError, match=msg):
                pd.Series(data, dtype=maindtype)
            return
        if maindtype is not None:
            series = pd.Series(data, dtype=maindtype)
        else:
            series = pd.Series(data)
        result = series.convert_dtypes(*params)
        param_names = ['infer_objects', 'convert_string', 'convert_integer', 'convert_boolean', 'convert_floating']
        params_dict = dict(zip(param_names, params))
        expected_dtype = expected_default
        for spec, dtype in expected_other.items():
            if all((params_dict[key] is val for key, val in zip(spec[::2], spec[1::2]))):
                expected_dtype = dtype
        if using_infer_string and expected_default == 'string' and (expected_dtype == object) and params[0] and (not params[1]):
            expected_dtype = 'string[pyarrow_numpy]'
        expected = pd.Series(data, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)
        copy = series.copy(deep=True)
        if result.notna().sum() > 0 and result.dtype in ['interval[int64, right]']:
            with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
                result[result.notna()] = np.nan
        else:
            result[result.notna()] = np.nan
        tm.assert_series_equal(series, copy)

    def test_convert_string_dtype(self, nullable_string_dtype):
        df = pd.DataFrame({'A': ['a', 'b', pd.NA], 'B': ['ä', 'ö', 'ü']}, dtype=nullable_string_dtype)
        result = df.convert_dtypes()
        tm.assert_frame_equal(df, result)

    def test_convert_bool_dtype(self):
        df = pd.DataFrame({'A': pd.array([True])})
        tm.assert_frame_equal(df, df.convert_dtypes())

    def test_convert_byte_string_dtype(self):
        byte_str = b'binary-string'
        df = pd.DataFrame(data={'A': byte_str}, index=[0])
        result = df.convert_dtypes()
        expected = df
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('infer_objects, dtype', [(True, 'Int64'), (False, 'object')])
    def test_convert_dtype_object_with_na(self, infer_objects, dtype):
        ser = pd.Series([1, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('infer_objects, dtype', [(True, 'Float64'), (False, 'object')])
    def test_convert_dtype_object_with_na_float(self, infer_objects, dtype):
        ser = pd.Series([1.5, pd.NA])
        result = ser.convert_dtypes(infer_objects=infer_objects)
        expected = pd.Series([1.5, pd.NA], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        pytest.importorskip('pyarrow')
        ser = pd.Series(range(2), dtype='int32[pyarrow]')
        result = ser.convert_dtypes(dtype_backend='numpy_nullable')
        expected = pd.Series(range(2), dtype='Int32')
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_pyarrow_null(self):
        pa = pytest.importorskip('pyarrow')
        ser = pd.Series([None, None])
        result = ser.convert_dtypes(dtype_backend='pyarrow')
        expected = pd.Series([None, None], dtype=pd.ArrowDtype(pa.null()))
        tm.assert_series_equal(result, expected)