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
@pytest.mark.xfail(reason='Api changed in 3.6.0')
@pytest.mark.slow
def test_plot_invalid_arg(self):
    df = DataFrame({'x': [1, 2], 'y': [3, 4]})
    msg = "'Line2D' object has no property 'blarg'"
    with pytest.raises(AttributeError, match=msg):
        df.plot.line(blarg=True)