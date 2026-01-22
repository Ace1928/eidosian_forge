import numpy as np
import pytest
from pandas.compat import (
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
def _compare_other(self, s, data, op, other):
    op_name = f'__{op.__name__}__'
    self.check_opname(s, op_name, other)