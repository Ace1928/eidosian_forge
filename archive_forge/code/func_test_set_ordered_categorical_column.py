from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def test_set_ordered_categorical_column():
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    mdf = pd.DataFrame(data)
    pdf = pandas.DataFrame(data)
    mdf['a'] = pd.Categorical(mdf['a'], ordered=True)
    pdf['a'] = pandas.Categorical(pdf['a'], ordered=True)
    df_equals(mdf, pdf)
    modin_categories = mdf['a'].dtype
    pandas_categories = pdf['a'].dtype
    assert modin_categories == pandas_categories