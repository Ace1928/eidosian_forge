from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
class TestArrowArray(base.ExtensionTests):

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, data[0])

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map(self, data_missing, na_action):
        if data_missing.dtype.kind in 'mM':
            result = data_missing.map(lambda x: x, na_action=na_action)
            expected = data_missing.to_numpy(dtype=object)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = data_missing.map(lambda x: x, na_action=na_action)
            if data_missing.dtype == 'float32[pyarrow]':
                expected = data_missing.to_numpy(dtype='float64', na_value=np.nan)
            else:
                expected = data_missing.to_numpy()
            tm.assert_numpy_array_equal(result, expected)

    def test_astype_str(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason=f'For {pa_dtype} .astype(str) decodes.'))
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None or pa.types.is_duration(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason='pd.Timestamp/pd.Timedelta repr different from numpy repr'))
        super().test_astype_str(data)

    @pytest.mark.parametrize('nullable_string_dtype', ['string[python]', pytest.param('string[pyarrow]', marks=td.skip_if_no('pyarrow'))])
    def test_astype_string(self, data, nullable_string_dtype, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None or pa.types.is_duration(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason='pd.Timestamp/pd.Timedelta repr different from numpy repr'))
        super().test_astype_string(data, nullable_string_dtype)

    def test_from_dtype(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f'pyarrow.type_for_alias cannot infer {pa_dtype}'
            request.applymarker(pytest.mark.xfail(reason=reason))
        super().test_from_dtype(data)

    def test_from_sequence_pa_array(self, data):
        result = type(data)._from_sequence(data._pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)
        result = type(data)._from_sequence(data._pa_array.combine_chunks(), dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

    def test_from_sequence_pa_array_notimplemented(self, request):
        with pytest.raises(NotImplementedError, match='Converting strings to'):
            ArrowExtensionArray._from_sequence_of_strings(['12-1'], dtype=pa.month_day_nano_interval())

    def test_from_sequence_of_strings_pa_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_time64(pa_dtype) and pa_dtype.equals('time64[ns]') and (not PY311):
            request.applymarker(pytest.mark.xfail(reason='Nanosecond time parsing not supported.'))
        elif pa_version_under11p0 and (pa.types.is_duration(pa_dtype) or pa.types.is_decimal(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowNotImplementedError, reason=f"pyarrow doesn't support parsing {pa_dtype}"))
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            _require_timezone_database(request)
        pa_array = data._pa_array.cast(pa.string())
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def check_accumulate(self, ser, op_name, skipna):
        result = getattr(ser, op_name)(skipna=skipna)
        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_type):
            if pa_type.bit_width == 32:
                int_type = 'int32[pyarrow]'
            else:
                int_type = 'int64[pyarrow]'
            ser = ser.astype(int_type)
            result = result.astype(int_type)
        result = result.astype('Float64')
        expected = getattr(ser.astype('Float64'), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type) or pa.types.is_decimal(pa_type):
            if op_name in ['cumsum', 'cumprod', 'cummax', 'cummin']:
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ['cumprod', 'cummax', 'cummin']:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == 'cumsum' and (not pa.types.is_duration(pa_type)):
                return False
            elif op_name == 'cumprod':
                return False
        return True

    @pytest.mark.parametrize('skipna', [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna, request):
        pa_type = data.dtype.pyarrow_dtype
        op_name = all_numeric_accumulations
        ser = pd.Series(data)
        if not self._supports_accumulation(ser, op_name):
            return super().test_accumulate_series(data, all_numeric_accumulations, skipna)
        if pa_version_under13p0 and all_numeric_accumulations != 'cumsum':
            opt = request.config.option
            if opt.markexpr and 'not slow' in opt.markexpr:
                pytest.skip(f'{all_numeric_accumulations} not implemented for pyarrow < 9')
            mark = pytest.mark.xfail(reason=f'{all_numeric_accumulations} not implemented for pyarrow < 9')
            request.applymarker(mark)
        elif all_numeric_accumulations == 'cumsum' and (pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)):
            request.applymarker(pytest.mark.xfail(reason=f'{all_numeric_accumulations} not implemented for {pa_type}', raises=NotImplementedError))
        self.check_accumulate(ser, op_name, skipna)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        dtype = ser.dtype
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_dtype) and op_name in ['sum', 'var', 'skew', 'kurt', 'prod']:
            if pa.types.is_duration(pa_dtype) and op_name in ['sum']:
                pass
            else:
                return False
        elif (pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)) and op_name in ['sum', 'mean', 'median', 'prod', 'std', 'sem', 'var', 'skew', 'kurt']:
            return False
        if pa.types.is_temporal(pa_dtype) and (not pa.types.is_duration(pa_dtype)) and (op_name in ['any', 'all']):
            return False
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        pa_dtype = ser.dtype.pyarrow_dtype
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            alt = ser.astype('Float64')
        else:
            alt = ser
        if op_name == 'count':
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        dtype = data.dtype
        pa_dtype = dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(raises=TypeError, reason=f'{all_numeric_reductions} is not implemented in pyarrow={pa.__version__} for {pa_dtype}')
        if all_numeric_reductions in {'skew', 'kurt'} and (dtype._is_numeric or dtype.kind == 'b'):
            request.applymarker(xfail_mark)
        elif pa.types.is_boolean(pa_dtype) and all_numeric_reductions in {'sem', 'std', 'var', 'median'}:
            request.applymarker(xfail_mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna, na_value, request):
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(raises=TypeError, reason=f'{all_boolean_reductions} is not implemented in pyarrow={pa.__version__} for {pa_dtype}')
        if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype):
            request.applymarker(xfail_mark)
        return super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        if op_name in ['max', 'min']:
            cmp_dtype = arr.dtype
        elif arr.dtype.name == 'decimal128(7, 3)[pyarrow]':
            if op_name not in ['median', 'var', 'std']:
                cmp_dtype = arr.dtype
            else:
                cmp_dtype = 'float64[pyarrow]'
        elif op_name in ['median', 'var', 'std', 'mean', 'skew']:
            cmp_dtype = 'float64[pyarrow]'
        else:
            cmp_dtype = {'i': 'int64[pyarrow]', 'u': 'uint64[pyarrow]', 'f': 'float64[pyarrow]'}[arr.dtype.kind]
        return cmp_dtype

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        op_name = all_numeric_reductions
        if op_name == 'skew':
            if data.dtype._is_numeric:
                mark = pytest.mark.xfail(reason='skew not implemented')
                request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize('typ', ['int64', 'uint64', 'float64'])
    def test_median_not_approximate(self, typ):
        result = pd.Series([1, 2], dtype=f'{typ}[pyarrow]').median()
        assert result == 1.5

    def test_in_numeric_groupby(self, data_for_grouping):
        dtype = data_for_grouping.dtype
        if is_string_dtype(dtype):
            df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1, 4], 'B': data_for_grouping, 'C': [1, 1, 1, 1, 1, 1, 1, 1]})
            expected = pd.Index(['C'])
            msg = re.escape(f'agg function failed [how->sum,dtype->{dtype}')
            with pytest.raises(TypeError, match=msg):
                df.groupby('A').sum()
            result = df.groupby('A').sum(numeric_only=True).columns
            tm.assert_index_equal(result, expected)
        else:
            super().test_in_numeric_groupby(data_for_grouping)

    def test_construct_from_string_own_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(pytest.mark.xfail(raises=NotImplementedError, reason=f'pyarrow.type_for_alias cannot infer {pa_dtype}'))
        if pa.types.is_string(pa_dtype):
            msg = 'string\\[pyarrow\\] should be constructed by StringDtype'
            with pytest.raises(TypeError, match=msg):
                dtype.construct_from_string(dtype.name)
            return
        super().test_construct_from_string_own_name(dtype)

    def test_is_dtype_from_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert not type(dtype).is_dtype(dtype.name)
        else:
            if pa.types.is_decimal(pa_dtype):
                request.applymarker(pytest.mark.xfail(raises=NotImplementedError, reason=f'pyarrow.type_for_alias cannot infer {pa_dtype}'))
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        msg = "'another_type' must end with '\\[pyarrow\\]'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string('another_type')

    def test_get_common_dtype(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_date(pa_dtype) or pa.types.is_time(pa_dtype) or (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None) or pa.types.is_binary(pa_dtype) or pa.types.is_decimal(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason=f'{pa_dtype} does not have associated numpy dtype findable by find_common_type'))
        super().test_get_common_dtype(dtype)

    def test_is_not_string_type(self, dtype):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert is_string_dtype(dtype)
        else:
            super().test_is_not_string_type(dtype)

    @pytest.mark.xfail(reason='GH 45419: pyarrow.ChunkedArray does not support views.', run=False)
    def test_view(self, data):
        super().test_view(data)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]
        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)
        result = data.fillna(method='backfill')
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    @pytest.mark.xfail(reason='GH 45419: pyarrow.ChunkedArray does not support views', run=False)
    def test_transpose(self, data):
        super().test_transpose(data)

    @pytest.mark.xfail(reason='GH 45419: pyarrow.ChunkedArray does not support views', run=False)
    def test_setitem_preserves_views(self, data):
        super().test_setitem_preserves_views(data)

    @pytest.mark.parametrize('dtype_backend', ['pyarrow', no_default])
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(self, engine, data, dtype_backend, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(pytest.mark.xfail(raises=NotImplementedError, reason=f'Parameterized types {pa_dtype} not supported.'))
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.unit in ('us', 'ns'):
            request.applymarker(pytest.mark.xfail(raises=ValueError, reason='https://github.com/pandas-dev/pandas/issues/49767'))
        elif pa.types.is_binary(pa_dtype):
            request.applymarker(pytest.mark.xfail(reason="CSV parsers don't correctly handle binary"))
        df = pd.DataFrame({'with_dtype': pd.Series(data, dtype=str(data.dtype))})
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        if pa.types.is_binary(pa_dtype):
            csv_output = BytesIO(csv_output)
        else:
            csv_output = StringIO(csv_output)
        result = pd.read_csv(csv_output, dtype={'with_dtype': str(data.dtype)}, engine=engine, dtype_backend=dtype_backend)
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_invert(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if not (pa.types.is_boolean(pa_dtype) or pa.types.is_integer(pa_dtype) or pa.types.is_string(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowNotImplementedError, reason=f'pyarrow.compute.invert does support {pa_dtype}'))
        if PY312 and pa.types.is_boolean(pa_dtype):
            with tm.assert_produces_warning(DeprecationWarning, match='Bitwise inversion', check_stacklevel=False):
                super().test_invert(data)
        else:
            super().test_invert(data)

    @pytest.mark.parametrize('periods', [1, -2])
    def test_diff(self, data, periods, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowInvalid, reason=f'diff with {pa_dtype} and periods={periods} will overflow'))
        super().test_diff(data, periods)

    def test_value_counts_returns_pyarrow_int64(self, data):
        data = data[:10]
        result = data.value_counts()
        assert result.dtype == ArrowDtype(pa.int64())
    _combine_le_expected_dtype = 'bool[pyarrow]'
    divmod_exc = NotImplementedError

    def get_op_from_name(self, op_name):
        short_opname = op_name.strip('_')
        if short_opname == 'rtruediv':

            def rtruediv(x, y):
                return np.divide(y, x)
            return rtruediv
        elif short_opname == 'rfloordiv':
            return lambda x, y: np.floor_divide(y, x)
        return tm.get_op_from_name(op_name)

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        expected = pointwise_result
        if op_name in ['eq', 'ne', 'lt', 'le', 'gt', 'ge']:
            return pointwise_result.astype('boolean[pyarrow]')
        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
            original_dtype = obj.iloc[:, 0].dtype
        else:
            expected_data = expected
            original_dtype = obj.dtype
        orig_pa_type = original_dtype.pyarrow_dtype
        if not was_frame and isinstance(other, pd.Series):
            if not (pa.types.is_floating(orig_pa_type) or (pa.types.is_integer(orig_pa_type) and op_name not in ['__truediv__', '__rtruediv__']) or pa.types.is_duration(orig_pa_type) or pa.types.is_timestamp(orig_pa_type) or pa.types.is_date(orig_pa_type) or pa.types.is_decimal(orig_pa_type)):
                return expected
        elif not (op_name == '__floordiv__' and pa.types.is_integer(orig_pa_type) or pa.types.is_duration(orig_pa_type) or pa.types.is_timestamp(orig_pa_type) or pa.types.is_date(orig_pa_type) or pa.types.is_decimal(orig_pa_type)):
            return expected
        pa_expected = pa.array(expected_data._values)
        if pa.types.is_duration(pa_expected.type):
            if pa.types.is_date(orig_pa_type):
                if pa.types.is_date64(orig_pa_type):
                    unit = 'ms'
                else:
                    unit = 's'
            else:
                unit = orig_pa_type.unit
                if type(other) in [datetime, timedelta] and unit in ['s', 'ms']:
                    unit = 'us'
            pa_expected = pa_expected.cast(f'duration[{unit}]')
        elif pa.types.is_decimal(pa_expected.type) and pa.types.is_decimal(orig_pa_type):
            alt = getattr(obj, op_name)(other)
            alt_dtype = tm.get_dtype(alt)
            assert isinstance(alt_dtype, ArrowDtype)
            if op_name == '__pow__' and isinstance(other, Decimal):
                alt_dtype = ArrowDtype(pa.float64())
            elif op_name == '__pow__' and isinstance(other, pd.Series) and (other.dtype == original_dtype):
                alt_dtype = ArrowDtype(pa.float64())
            else:
                assert pa.types.is_decimal(alt_dtype.pyarrow_dtype)
            return expected.astype(alt_dtype)
        else:
            pa_expected = pa_expected.cast(orig_pa_type)
        pd_expected = type(expected_data._values)(pa_expected)
        if was_frame:
            expected = pd.DataFrame(pd_expected, index=expected.index, columns=expected.columns)
        else:
            expected = pd.Series(pd_expected)
        return expected

    def _is_temporal_supported(self, opname, pa_dtype):
        return (opname in ('__add__', '__radd__') or (opname in ('__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__') and (not pa_version_under14p0))) and pa.types.is_duration(pa_dtype) or (opname in ('__sub__', '__rsub__') and pa.types.is_temporal(pa_dtype))

    def _get_expected_exception(self, op_name: str, obj, other) -> type[Exception] | None:
        if op_name in ('__divmod__', '__rdivmod__'):
            return self.divmod_exc
        dtype = tm.get_dtype(obj)
        pa_dtype = dtype.pyarrow_dtype
        arrow_temporal_supported = self._is_temporal_supported(op_name, pa_dtype)
        if op_name in {'__mod__', '__rmod__'}:
            exc = NotImplementedError
        elif arrow_temporal_supported:
            exc = None
        elif op_name in ['__add__', '__radd__'] and (pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)):
            exc = None
        elif not (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)):
            exc = pa.ArrowNotImplementedError
        else:
            exc = None
        return exc

    def _get_arith_xfail_marker(self, opname, pa_dtype):
        mark = None
        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)
        if opname == '__rpow__' and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)):
            mark = pytest.mark.xfail(reason=f'GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL for {pa_dtype}')
        elif arrow_temporal_supported and (pa.types.is_time(pa_dtype) or (opname in ('__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__') and pa.types.is_duration(pa_dtype))):
            mark = pytest.mark.xfail(raises=TypeError, reason=f'{opname} not supported betweenpd.NA and {pa_dtype} Python scalar')
        elif opname == '__rfloordiv__' and (pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)):
            mark = pytest.mark.xfail(raises=pa.ArrowInvalid, reason='divide by 0')
        elif opname == '__rtruediv__' and pa.types.is_decimal(pa_dtype):
            mark = pytest.mark.xfail(raises=pa.ArrowInvalid, reason='divide by 0')
        return mark

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators == '__rmod__' and pa.types.is_binary(pa_dtype):
            pytest.skip('Skip testing Python string formatting')
        elif all_arithmetic_operators in ('__rmul__', '__mul__') and (pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=TypeError, reason='Can only string multiply by an integer.'))
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators == '__rmod__' and (pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)):
            pytest.skip('Skip testing Python string formatting')
        elif all_arithmetic_operators in ('__rmul__', '__mul__') and (pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=TypeError, reason='Can only string multiply by an integer.'))
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if all_arithmetic_operators in ('__sub__', '__rsub__') and pa.types.is_unsigned_integer(pa_dtype):
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowInvalid, reason=f'Implemented pyarrow.compute.subtract_checked which raises on overflow for {pa_dtype}'))
        elif all_arithmetic_operators in ('__rmul__', '__mul__') and (pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)):
            request.applymarker(pytest.mark.xfail(raises=TypeError, reason='Can only string multiply by an integer.'))
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))
        self.check_opname(ser, op_name, other)

    def test_add_series_with_extension_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa_dtype.equals('int8'):
            request.applymarker(pytest.mark.xfail(raises=pa.ArrowInvalid, reason=f'raises on overflow for {pa_dtype}'))
        super().test_add_series_with_extension_array(data)

    def test_invalid_other_comp(self, data, comparison_op):
        with pytest.raises(NotImplementedError, match=".* not implemented for <class 'object'>"):
            comparison_op(data, object())

    @pytest.mark.parametrize('masked_dtype', ['boolean', 'Int64', 'Float64'])
    def test_comp_masked_numpy(self, masked_dtype, comparison_op):
        data = [1, 0, None]
        ser_masked = pd.Series(data, dtype=masked_dtype)
        ser_pa = pd.Series(data, dtype=f'{masked_dtype.lower()}[pyarrow]')
        result = comparison_op(ser_pa, ser_masked)
        if comparison_op in [operator.lt, operator.gt, operator.ne]:
            exp = [False, False, None]
        else:
            exp = [True, True, None]
        expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)