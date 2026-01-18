from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_scatter_unknown_colormap(self):
    df = DataFrame({'a': [1, 2, 3], 'b': 4})
    with pytest.raises((ValueError, KeyError), match="'unknown' is not a"):
        df.plot(x='a', y='b', colormap='unknown', kind='scatter')