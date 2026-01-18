from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def unstack_and_compare(df, column_name):
    unstacked1 = df.unstack([column_name])
    unstacked2 = df.unstack(column_name)
    tm.assert_frame_equal(unstacked1, unstacked2)