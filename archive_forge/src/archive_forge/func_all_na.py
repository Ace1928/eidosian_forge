import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def all_na(x):
    return x.isnull().all().all()