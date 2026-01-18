from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets

Tests for the following offsets:
- CustomBusinessMonthBase
- CustomBusinessMonthBegin
- CustomBusinessMonthEnd
