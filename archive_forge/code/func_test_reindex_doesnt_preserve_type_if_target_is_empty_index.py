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
@pytest.mark.parametrize('labels,dtype', [(DatetimeIndex([]), np.datetime64)])
def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self, labels, dtype):
    index = Index(list('abc'))
    assert index.reindex(labels)[0].dtype.type == dtype