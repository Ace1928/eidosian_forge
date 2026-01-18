import io
import numpy as np
import pytest
from pandas import (
def test_numerics():
    data = DataFrame([[1, 'a'], [2, 'b']])
    result = data.style.bar()._compute().ctx
    assert (0, 1) not in result
    assert (1, 1) not in result