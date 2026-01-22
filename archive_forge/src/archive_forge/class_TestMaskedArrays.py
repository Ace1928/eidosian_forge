import warnings
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
class TestMaskedArrays(base.ExtensionTests):

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map(self, data_missing, na_action):
        result = data_missing.map(lambda x: x, na_action=na_action)
        if data_missing.dtype == Float32Dtype():
            expected = data_missing.to_numpy(dtype='float64', na_value=np.nan)
        else:
            expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_map_na_action_ignore(self, data_missing_for_sorting):
        zero = data_missing_for_sorting[2]
        result = data_missing_for_sorting.map(lambda x: zero, na_action='ignore')
        if data_missing_for_sorting.dtype.kind == 'b':
            expected = np.array([False, pd.NA, False], dtype=object)
        else:
            expected = np.array([zero, np.nan, zero])
        tm.assert_numpy_array_equal(result, expected)

    def _get_expected_exception(self, op_name, obj, other):
        try:
            dtype = tm.get_dtype(obj)
        except AttributeError:
            dtype = tm.get_dtype(other)
        if dtype.kind == 'b':
            if op_name.strip('_').lstrip('r') in ['pow', 'truediv', 'floordiv']:
                return NotImplementedError
            elif op_name in ['__sub__', '__rsub__']:
                return TypeError
            return None
        return None

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        sdtype = tm.get_dtype(obj)
        expected = pointwise_result
        if op_name in ('eq', 'ne', 'le', 'ge', 'lt', 'gt'):
            return expected.astype('boolean')
        if sdtype.kind in 'iu':
            if op_name in ('__rtruediv__', '__truediv__', '__div__'):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Downcasting object dtype arrays', category=FutureWarning)
                    filled = expected.fillna(np.nan)
                expected = filled.astype('Float64')
            else:
                expected = expected.astype(sdtype)
        elif sdtype.kind == 'b':
            if op_name in ('__floordiv__', '__rfloordiv__', '__pow__', '__rpow__', '__mod__', '__rmod__'):
                expected = expected.astype('Int8')
            elif op_name in ('__truediv__', '__rtruediv__'):
                op = self.get_op_from_name(op_name)
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype('Float64')
            if op_name == '__rpow__':
                result = getattr(obj, op_name)(other)
                expected[result.isna()] = np.nan
        else:
            expected = expected.astype(sdtype)
        return expected

    def test_divmod_series_array(self, data, data_for_twos, request):
        if data.dtype.kind == 'b':
            mark = pytest.mark.xfail(reason='Inconsistency between floordiv and divmod; we raise for floordiv but not for divmod. This matches what we do for non-masked bool dtype.')
            request.applymarker(mark)
        super().test_divmod_series_array(data, data_for_twos)

    def test_combine_le(self, data_repeated):
        orig_data1, orig_data2 = data_repeated(2)
        if orig_data1.dtype.kind == 'b':
            self._combine_le_expected_dtype = 'boolean'
        else:
            self._combine_le_expected_dtype = object
        super().test_combine_le(data_repeated)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ['any', 'all'] and ser.dtype.kind != 'b':
            pytest.skip(reason='Tested in tests/reductions/test_reductions.py')
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        cmp_dtype = 'int64'
        if ser.dtype.kind == 'f':
            cmp_dtype = ser.dtype.numpy_dtype
        elif ser.dtype.kind == 'b':
            if op_name in ['min', 'max']:
                cmp_dtype = 'bool'
        alt = ser.dropna().astype(cmp_dtype)
        if op_name == 'count':
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
            if not skipna and ser.isna().any() and (op_name not in ['any', 'all']):
                expected = pd.NA
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        if is_float_dtype(arr.dtype):
            cmp_dtype = arr.dtype.name
        elif op_name in ['mean', 'median', 'var', 'std', 'skew']:
            cmp_dtype = 'Float64'
        elif op_name in ['max', 'min']:
            cmp_dtype = arr.dtype.name
        elif arr.dtype in ['Int64', 'UInt64']:
            cmp_dtype = arr.dtype.name
        elif is_signed_integer_dtype(arr.dtype):
            cmp_dtype = 'Int32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'Int64'
        elif is_unsigned_integer_dtype(arr.dtype):
            cmp_dtype = 'UInt32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'UInt64'
        elif arr.dtype.kind == 'b':
            if op_name in ['mean', 'median', 'var', 'std', 'skew']:
                cmp_dtype = 'Float64'
            elif op_name in ['min', 'max']:
                cmp_dtype = 'boolean'
            elif op_name in ['sum', 'prod']:
                cmp_dtype = 'Int32' if is_platform_windows() and (not np_version_gt2 or not skipna) or not IS64 else 'Int64'
            else:
                raise TypeError('not supposed to reach this')
        else:
            raise TypeError('not supposed to reach this')
        return cmp_dtype

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        length = 64
        if is_windows_or_32bit:
            if not ser.dtype.itemsize == 8:
                length = 32
        if ser.dtype.name.startswith('U'):
            expected_dtype = f'UInt{length}'
        elif ser.dtype.name.startswith('I'):
            expected_dtype = f'Int{length}'
        elif ser.dtype.name.startswith('F'):
            expected_dtype = ser.dtype
        elif ser.dtype.kind == 'b':
            if op_name in ('cummin', 'cummax'):
                expected_dtype = 'boolean'
            else:
                expected_dtype = f'Int{length}'
        if expected_dtype == 'Float32' and op_name == 'cumprod' and skipna:
            pytest.skip(f'Float32 precision lead to large differences with op {op_name} and skipna={skipna}')
        if op_name == 'cumsum':
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser.astype('float64'), op_name)(skipna=skipna), dtype=expected_dtype))
            tm.assert_series_equal(result, expected)
        elif op_name in ['cummax', 'cummin']:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser.astype('float64'), op_name)(skipna=skipna), dtype=ser.dtype))
            tm.assert_series_equal(result, expected)
        elif op_name == 'cumprod':
            result = getattr(ser[:12], op_name)(skipna=skipna)
            expected = pd.Series(pd.array(getattr(ser[:12].astype('float64'), op_name)(skipna=skipna), dtype=expected_dtype))
            tm.assert_series_equal(result, expected)
        else:
            raise NotImplementedError(f'{op_name} not supported')