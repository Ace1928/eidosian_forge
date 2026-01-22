import pytest
import pandas as pd
import pandas._testing as tm
def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
    return False