import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base

    Groupby-specific tests are overridden because boolean only has 2
    unique values, base tests uses 3 groups.
    