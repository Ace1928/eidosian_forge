from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_bool_fails(self, capsys):
    s = Series([DataFrame(np.random.randn(2, 2)) for i in range(5)])
    repr(s)
    captured = capsys.readouterr()
    assert captured.err == ''