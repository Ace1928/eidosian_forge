from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key', [{1}, {1: 1}, ({1}, 'a'), ({1: 1}, 'a'), (1, {'a'}), (1, {'a': 'a'})])
def test_setitem_dict_and_set_disallowed(self, key):
    df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    with pytest.raises(TypeError, match='as an indexer is not supported'):
        df.loc[key] = 1