from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def _check_merge(x, y):
    for how in ['inner', 'left', 'outer']:
        for sort in [True, False]:
            result = x.join(y, how=how, sort=sort)
            expected = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index('index')
            tm.assert_frame_equal(result, expected, check_names=False)