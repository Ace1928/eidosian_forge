from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
@pytest.mark.parametrize('skipna', [True, False])
def test_reduce_frame(self, data, all_numeric_reductions, skipna):
    op_name = all_numeric_reductions
    ser = pd.Series(data)
    if not is_numeric_dtype(ser.dtype):
        pytest.skip(f'{ser.dtype} is not numeric dtype')
    if op_name in ['count', 'kurt', 'sem']:
        pytest.skip(f'{op_name} not an array method')
    if not self._supports_reduction(ser, op_name):
        pytest.skip(f'Reduction {op_name} not supported for this dtype')
    self.check_reduce_frame(ser, op_name, skipna)