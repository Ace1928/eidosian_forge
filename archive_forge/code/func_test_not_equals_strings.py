from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_not_equals_strings(self, index):
    other = Index([str(x) for x in index], dtype=object)
    assert not index.equals(other)
    assert not index.equals(CategoricalIndex(other))