from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
class SubclassedDataFrame(DataFrame):
    _metadata = ['my_extra_data']

    def __init__(self, my_extra_data, *args, **kwargs) -> None:
        self.my_extra_data = my_extra_data
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return functools.partial(type(self), self.my_extra_data)

    @property
    def _constructor_sliced(self):
        return SubclassedSeries