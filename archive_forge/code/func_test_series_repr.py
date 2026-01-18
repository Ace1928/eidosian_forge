import io
import pytest
import pandas as pd
def test_series_repr(self, data):
    ser = pd.Series(data)
    assert data.dtype.name in repr(ser)