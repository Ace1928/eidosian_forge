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
@pytest.mark.parametrize('name', [None, 'foobar'])
@pytest.mark.parametrize('labels', [[], np.array([]), ['A', 'B', 'C'], ['C', 'B', 'A'], np.array(['A', 'B', 'C']), np.array(['C', 'B', 'A']), date_range('20130101', periods=3).values, date_range('20130101', periods=3).tolist()])
def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name, labels):
    index = Index([0, 1, 2])
    index.name = name
    assert index.reindex(labels)[0].name == name