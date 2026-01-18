import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
@pytest.mark.parametrize('obj', [(obj,) for obj in pd.__dict__.values() if callable(obj)])
def test_serializable(obj):
    unpickled = tm.round_trip_pickle(obj)
    assert type(obj) == type(unpickled)