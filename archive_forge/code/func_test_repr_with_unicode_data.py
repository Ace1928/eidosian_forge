import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_with_unicode_data():
    with pd.option_context('display.encoding', 'UTF-8'):
        d = {'a': ['א', 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
        index = pd.DataFrame(d).set_index(['a', 'b']).index
        assert '\\' not in repr(index)