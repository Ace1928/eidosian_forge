import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED

        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        