from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_tab_complete_ipython6_warning(ip):
    from IPython.core.completer import provisionalcompleter
    code = dedent('    import numpy as np\n    from pandas import Series, date_range\n    data = np.arange(10, dtype=np.float64)\n    index = date_range("2020-01-01", periods=len(data))\n    s = Series(data, index=index)\n    rs = s.resample("D")\n    ')
    ip.run_cell(code)
    with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
        with provisionalcompleter('ignore'):
            list(ip.Completer.completions('rs.', 1))