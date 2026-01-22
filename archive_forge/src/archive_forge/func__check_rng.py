from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_rng(rng):
    converted = rng.to_pydatetime()
    assert isinstance(converted, np.ndarray)
    for x, stamp in zip(converted, rng):
        assert isinstance(x, datetime)
        assert x == stamp.to_pydatetime()
        assert x.tzinfo == stamp.tzinfo