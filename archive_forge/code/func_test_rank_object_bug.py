from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_rank_object_bug(self):
    Series([np.nan] * 32).astype(object).rank(ascending=True)
    Series([np.nan] * 32).astype(object).rank(ascending=False)