from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
@pytest.fixture(params=[operator.add, ops.radd, operator.sub, ops.rsub, operator.mul, ops.rmul, operator.truediv, ops.rtruediv, operator.floordiv, ops.rfloordiv, operator.mod, ops.rmod, operator.pow, ops.rpow, operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge, operator.and_, ops.rand_, operator.xor, ops.rxor, operator.or_, ops.ror_])
def all_binary_operators(request):
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.
    """
    return request.param