import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
class NonCommutativeMultiplyInteger:
    """int-like class with non-commutative multiply operation.

    We need to test that rmul and mul do different things even when
    multiplication is not commutative, but almost all multiplication is
    commutative. This class' fake multiplication overloads are not commutative
    when you multiply an instance of this class with pandas.series, which
    does not know how to __mul__ with this class. e.g.

    NonCommutativeMultiplyInteger(2) * pd.Series(1, dtype=int) == pd.Series(2, dtype=int)
    pd.Series(1, dtype=int) * NonCommutativeMultiplyInteger(2) == pd.Series(3, dtype=int)
    """

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f'must initialize with integer, but got {value} of type {type(value)}')
        self.value = value

    def __mul__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return self.value * other

    def __rmul__(self, other):
        return self.value * other + 1