from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_reindex_ignoring_level(self):
    idx = Index([1, 2, 3], name='x')
    idx2 = Index([1, 2, 3, 4], name='x')
    expected = Index([1, 2, 3, 4], name='x')
    result, _ = idx.reindex(idx2, level='x')
    tm.assert_index_equal(result, expected)