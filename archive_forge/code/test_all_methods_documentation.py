import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

Tests that apply to all groupby operation methods.

The only tests that should appear here are those that use the `groupby_func` fixture.
Even if it does use that fixture, prefer a more specific test file if it available
such as:

 - test_categorical
 - test_groupby_dropna
 - test_groupby_subclass
 - test_raises
