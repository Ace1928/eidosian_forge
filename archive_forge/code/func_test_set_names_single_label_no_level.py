from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail
def test_set_names_single_label_no_level(self, index_flat):
    with pytest.raises(TypeError, match='list-like'):
        index_flat.set_names('a')