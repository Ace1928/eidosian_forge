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
def _check_errorbar_color(containers, expected, has_err='has_xerr'):
    lines = []
    errs = next((c.lines for c in ax.containers if getattr(c, has_err, False)))
    for el in errs:
        if is_list_like(el):
            lines.extend(el)
        else:
            lines.append(el)
    err_lines = [x for x in lines if x in ax.collections]
    _check_colors(err_lines, linecolors=np.array([expected] * len(err_lines)))