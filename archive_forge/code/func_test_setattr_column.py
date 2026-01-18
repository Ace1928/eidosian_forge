from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setattr_column(self):
    df = DataFrame({'foobar': 1}, index=range(10))
    df.foobar = 5
    assert (df.foobar == 5).all()