from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
shared endpoints are marked as overlapping