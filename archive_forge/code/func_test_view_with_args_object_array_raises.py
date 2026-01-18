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
@pytest.mark.parametrize('index', ['string', pytest.param('categorical', marks=pytest.mark.xfail(reason='gh-25464')), 'bool-object', 'bool-dtype', 'empty'], indirect=True)
def test_view_with_args_object_array_raises(self, index):
    if index.dtype == bool:
        msg = 'When changing to a larger dtype'
        with pytest.raises(ValueError, match=msg):
            index.view('i8')
    elif index.dtype == 'string':
        with pytest.raises(NotImplementedError, match='i8'):
            index.view('i8')
    else:
        msg = 'Cannot change data-type for object array'
        with pytest.raises(TypeError, match=msg):
            index.view('i8')