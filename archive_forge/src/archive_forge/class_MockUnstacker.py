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
class MockUnstacker(reshape_lib._Unstacker):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise Exception("Don't compute final result.")