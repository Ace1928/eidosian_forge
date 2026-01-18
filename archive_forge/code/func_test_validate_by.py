import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def test_validate_by():
    """Test ``modin.core.dataframe.algebra.default2pandas.groupby.GroupBy.validate_by``."""

    def compare(obj1, obj2):
        assert type(obj1) is type(obj2), f'Both objects must be instances of the same type: {type(obj1)} != {type(obj2)}.'
        if isinstance(obj1, list):
            for val1, val2 in itertools.zip_longest(obj1, obj2):
                df_equals(val1, val2)
        else:
            df_equals(obj1, obj2)
    reduced_frame = pandas.DataFrame({MODIN_UNNAMED_SERIES_LABEL: [1, 2, 3]})
    series_result = GroupBy.validate_by(reduced_frame)
    series_reference = [pandas.Series([1, 2, 3], name=None)]
    compare(series_reference, series_result)
    splited_df = [pandas.Series([1, 2, 3], name=f'col{i}') for i in range(3)]
    splited_df_result = GroupBy.validate_by(pandas.concat(splited_df, axis=1, copy=True))
    compare(splited_df, splited_df_result)
    by = ['col1', 'col2', pandas.DataFrame({MODIN_UNNAMED_SERIES_LABEL: [1, 2, 3]})]
    result_by = GroupBy.validate_by(by)
    reference_by = ['col1', 'col2', pandas.Series([1, 2, 3], name=None)]
    compare(reference_by, result_by)