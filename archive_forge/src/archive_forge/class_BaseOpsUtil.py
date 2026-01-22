from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
class BaseOpsUtil:
    series_scalar_exc: type[Exception] | None = TypeError
    frame_scalar_exc: type[Exception] | None = TypeError
    series_array_exc: type[Exception] | None = TypeError
    divmod_exc: type[Exception] | None = TypeError

    def _get_expected_exception(self, op_name: str, obj, other) -> type[Exception] | None:
        if op_name in ['__divmod__', '__rdivmod__']:
            result = self.divmod_exc
        elif isinstance(obj, pd.Series) and isinstance(other, pd.Series):
            result = self.series_array_exc
        elif isinstance(obj, pd.Series):
            result = self.series_scalar_exc
        else:
            result = self.frame_scalar_exc
        if using_pyarrow_string_dtype() and result is not None:
            import pyarrow as pa
            result = (result, pa.lib.ArrowNotImplementedError, NotImplementedError)
        return result

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        return pointwise_result

    def get_op_from_name(self, op_name: str):
        return tm.get_op_from_name(op_name)

    @final
    def check_opname(self, ser: pd.Series, op_name: str, other):
        exc = self._get_expected_exception(op_name, ser, other)
        op = self.get_op_from_name(op_name)
        self._check_op(ser, op, other, op_name, exc)

    @final
    def _combine(self, obj, other, op):
        if isinstance(obj, pd.DataFrame):
            if len(obj.columns) != 1:
                raise NotImplementedError
            expected = obj.iloc[:, 0].combine(other, op).to_frame()
        else:
            expected = obj.combine(other, op)
        return expected

    @final
    def _check_op(self, ser: pd.Series, op, other, op_name: str, exc=NotImplementedError):
        if exc is None:
            result = op(ser, other)
            expected = self._combine(ser, other, op)
            expected = self._cast_pointwise_result(op_name, ser, other, expected)
            assert isinstance(result, type(ser))
            tm.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(ser, other)

    @final
    def _check_divmod_op(self, ser: pd.Series, op, other):
        if op is divmod:
            exc = self._get_expected_exception('__divmod__', ser, other)
        else:
            exc = self._get_expected_exception('__rdivmod__', ser, other)
        if exc is None:
            result_div, result_mod = op(ser, other)
            if op is divmod:
                expected_div, expected_mod = (ser // other, ser % other)
            else:
                expected_div, expected_mod = (other // ser, other % ser)
            tm.assert_series_equal(result_div, expected_div)
            tm.assert_series_equal(result_mod, expected_mod)
        else:
            with pytest.raises(exc):
                divmod(ser, other)