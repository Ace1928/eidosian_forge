from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_not_equals_numeric(self, index):
    assert not index.equals(Index(index.asi8))
    assert not index.equals(Index(index.asi8.astype('u8')))
    assert not index.equals(Index(index.asi8).astype('f8'))