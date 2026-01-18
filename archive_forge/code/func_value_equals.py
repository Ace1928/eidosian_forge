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
def value_equals(obj1, obj2):
    """Check wherher two scalar or list-like values are equal and raise an ``AssertionError`` if they aren't."""
    if is_list_like(obj1):
        np.testing.assert_array_equal(obj1, obj2)
    else:
        assert obj1 == obj2 or (np.isnan(obj1) and np.isnan(obj2))