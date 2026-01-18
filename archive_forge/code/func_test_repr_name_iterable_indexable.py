from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_name_iterable_indexable(self):
    s = Series([1, 2, 3], name=np.int64(3))
    repr(s)
    s.name = ('א',) * 2
    repr(s)