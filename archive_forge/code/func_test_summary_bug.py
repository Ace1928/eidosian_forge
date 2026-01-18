import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
def test_summary_bug(self):
    ind = Index(['{other}%s', '~:{range}:0'], name='A')
    result = ind._summary()
    assert '~:{range}:0' in result
    assert '{other}%s' in result