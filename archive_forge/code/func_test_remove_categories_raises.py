import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
@pytest.mark.parametrize('removals', [['c'], ['c', np.nan], 'c', ['c', 'c']])
def test_remove_categories_raises(self, removals):
    cat = Categorical(['a', 'b', 'a'])
    message = re.escape("removals must all be in old categories: {'c'}")
    with pytest.raises(ValueError, match=message):
        cat.remove_categories(removals)