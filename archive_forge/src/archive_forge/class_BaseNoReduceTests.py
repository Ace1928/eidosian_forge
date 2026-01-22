from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
class BaseNoReduceTests(BaseReduceTests):
    """we don't define any reductions"""