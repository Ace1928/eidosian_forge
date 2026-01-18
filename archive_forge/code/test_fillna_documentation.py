import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops

Though Index.fillna and Series.fillna has separate impl,
test here to confirm these works as the same
