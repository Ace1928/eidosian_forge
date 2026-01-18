from __future__ import annotations
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import doc
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import (
from pandas.core.util.numba_ import (
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.window.online import (
from pandas.core.window.rolling import (
def var_func(values, begin, end, min_periods):
    return wfunc(values, begin, end, min_periods, values)