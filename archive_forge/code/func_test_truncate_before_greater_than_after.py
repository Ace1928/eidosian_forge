import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_truncate_before_greater_than_after():
    df = pd.DataFrame([[1, 2, 3]])
    with pytest.raises(ValueError, match='Truncate: 1 must be after 2'):
        df.truncate(before=2, after=1)