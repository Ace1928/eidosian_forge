from __future__ import annotations
import numpy as np
import pytest
from xarray.core import dtypes
def test_result_type_scalar() -> None:
    actual = dtypes.result_type(np.arange(3, dtype=np.float32), np.nan)
    assert actual == np.float32