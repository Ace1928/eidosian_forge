import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_negative(self, data):
    n = len(data)
    result = data.take([0, -n, n - 1, -1])
    expected = data.take([0, 0, n - 1, n - 1])
    tm.assert_extension_array_equal(result, expected)