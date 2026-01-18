from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tidy_repr(self):
    a = Series(['א'] * 1000)
    a.name = 'title1'
    repr(a)