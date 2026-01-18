import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_subclass_empty_repr(self):
    sub_series = tm.SubclassedSeries()
    assert 'SubclassedSeries' in repr(sub_series)