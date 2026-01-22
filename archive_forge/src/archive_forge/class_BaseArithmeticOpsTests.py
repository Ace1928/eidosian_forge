from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
class BaseArithmeticOpsTests(BaseOpsUtil):
    """
    Various Series and DataFrame arithmetic ops methods.

    Subclasses supporting various ops should set the class variables
    to indicate that they support ops of that kind

    * series_scalar_exc = TypeError
    * frame_scalar_exc = TypeError
    * series_array_exc = TypeError
    * divmod_exc = TypeError
    """
    series_scalar_exc: type[Exception] | None = TypeError
    frame_scalar_exc: type[Exception] | None = TypeError
    series_array_exc: type[Exception] | None = TypeError
    divmod_exc: type[Exception] | None = TypeError

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        if all_arithmetic_operators == '__rmod__' and is_string_dtype(data.dtype):
            pytest.skip('Skip testing Python string formatting')
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, ser.iloc[0])

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        if all_arithmetic_operators == '__rmod__' and is_string_dtype(data.dtype):
            pytest.skip('Skip testing Python string formatting')
        op_name = all_arithmetic_operators
        df = pd.DataFrame({'A': data})
        self.check_opname(df, op_name, data[0])

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)))

    def test_divmod(self, data):
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, 1)
        self._check_divmod_op(1, ops.rdivmod, ser)

    def test_divmod_series_array(self, data, data_for_twos):
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data)
        other = data_for_twos
        self._check_divmod_op(other, ops.rdivmod, ser)
        other = pd.Series(other)
        self._check_divmod_op(other, ops.rdivmod, ser)

    def test_add_series_with_extension_array(self, data):
        ser = pd.Series(data)
        exc = self._get_expected_exception('__add__', ser, data)
        if exc is not None:
            with pytest.raises(exc):
                ser + data
            return
        result = ser + data
        expected = pd.Series(data + data)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('box', [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize('op_name', [x for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods if not x.startswith('__r')])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data, box, op_name):
        other = box(data)
        if hasattr(data, op_name):
            result = getattr(data, op_name)(other)
            assert result is NotImplemented