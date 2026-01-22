from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
class CustomIterator1:

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        try:
            return {0: df1, 1: df2}[index]
        except KeyError as err:
            raise IndexError from err