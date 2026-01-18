import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
Test frequency conversion of date objects