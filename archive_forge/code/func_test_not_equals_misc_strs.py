from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_not_equals_misc_strs(self, index):
    other = Index(list('abc'))
    assert not index.equals(other)