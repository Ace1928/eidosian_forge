from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _get_with_delta(delta, freq='YE-DEC'):
    return date_range(to_datetime('1/1/2001') + delta, to_datetime('12/31/2009') + delta, freq=freq)