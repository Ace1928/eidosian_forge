import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_subplots_constrained_layout(self):
    idx = date_range(start='now', periods=10)
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
    kwargs = {}
    if hasattr(mpl.pyplot.Figure, 'get_constrained_layout'):
        kwargs['constrained_layout'] = True
    _, axes = mpl.pyplot.subplots(2, **kwargs)
    with tm.assert_produces_warning(None):
        df.plot(ax=axes[0])
        with tm.ensure_clean(return_filelike=True) as path:
            mpl.pyplot.savefig(path)