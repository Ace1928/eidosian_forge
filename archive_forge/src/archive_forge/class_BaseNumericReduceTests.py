from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
class BaseNumericReduceTests(BaseReduceTests):

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ['any', 'all']:
            pytest.skip('These are tested in BaseBooleanReduceTests')
        return True