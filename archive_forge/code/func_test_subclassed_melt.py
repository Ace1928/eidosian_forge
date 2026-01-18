import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclassed_melt(self):
    cheese = tm.SubclassedDataFrame({'first': ['John', 'Mary'], 'last': ['Doe', 'Bo'], 'height': [5.5, 6.0], 'weight': [130, 150]})
    melted = pd.melt(cheese, id_vars=['first', 'last'])
    expected = tm.SubclassedDataFrame([['John', 'Doe', 'height', 5.5], ['Mary', 'Bo', 'height', 6.0], ['John', 'Doe', 'weight', 130], ['Mary', 'Bo', 'weight', 150]], columns=['first', 'last', 'variable', 'value'])
    tm.assert_frame_equal(melted, expected)