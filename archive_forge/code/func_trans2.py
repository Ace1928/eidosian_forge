from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def trans2(group):
    grouped = group.groupby(df.reindex(group.index)['B'])
    return grouped.sum().sort_values().iloc[:2]