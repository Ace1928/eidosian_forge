import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_top_level_method(self, df):
    result = melt(df)
    assert result.columns.tolist() == ['variable', 'value']