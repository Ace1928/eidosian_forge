import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
class TestArithmeticOps(base.BaseArithmeticOpsTests):
    implements = {'__sub__', '__rsub__'}

    def check_opname(self, s, op_name, other, exc=None):
        exc = None
        if op_name.strip('_').lstrip('r') in ['pow', 'truediv', 'floordiv']:
            exc = NotImplementedError
        super().check_opname(s, op_name, other, exc=exc)

    def _check_op(self, obj, op, other, op_name, exc=NotImplementedError):
        if exc is None:
            if op_name in self.implements:
                msg = 'numpy boolean subtract'
                with pytest.raises(TypeError, match=msg):
                    op(obj, other)
                return
            result = op(obj, other)
            expected = self._combine(obj, other, op)
            if op_name in ('__floordiv__', '__rfloordiv__', '__pow__', '__rpow__', '__mod__', '__rmod__'):
                expected = expected.astype('Int8')
            elif op_name in ('__truediv__', '__rtruediv__'):
                expected = self._combine(obj.astype(float), other, op)
                expected = expected.astype('Float64')
            if op_name == '__rpow__':
                expected[result.isna()] = np.nan
            self.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(obj, other)

    @pytest.mark.xfail(reason='Inconsistency between floordiv and divmod; we raise for floordiv but not for divmod. This matches what we do for non-masked bool dtype.')
    def test_divmod_series_array(self, data, data_for_twos):
        super().test_divmod_series_array(data, data_for_twos)

    @pytest.mark.xfail(reason='Inconsistency between floordiv and divmod; we raise for floordiv but not for divmod. This matches what we do for non-masked bool dtype.')
    def test_divmod(self, data):
        super().test_divmod(data)