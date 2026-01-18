import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm

    Fixture for DataFrame of ints with date_range index

    Columns are ['A', 'B'].
    