import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('func', [operator.itemgetter(['a']), operator.methodcaller('add', 1), operator.methodcaller('rename', str.upper), operator.methodcaller('rename', 'name'), operator.methodcaller('abs'), np.abs])
def test_preserved_series(self, func):
    s = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
    assert func(s).flags.allows_duplicate_labels is False