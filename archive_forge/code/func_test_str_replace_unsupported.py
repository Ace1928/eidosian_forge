from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('arg_name, arg', [['pat', re.compile('b')], ['repl', str], ['case', False], ['flags', 1]])
def test_str_replace_unsupported(arg_name, arg):
    ser = pd.Series(['abc', None], dtype=ArrowDtype(pa.string()))
    kwargs = {'pat': 'b', 'repl': 'x', 'regex': True}
    kwargs[arg_name] = arg
    with pytest.raises(NotImplementedError, match='replace is not supported'):
        ser.str.replace(**kwargs)