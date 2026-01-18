from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_tab_complete_warning(self, ip):
    pytest.importorskip('IPython', minversion='6.0.0')
    from IPython.core.completer import provisionalcompleter
    code = 'import pandas as pd; idx = pd.Index([1, 2])'
    ip.run_cell(code)
    with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
        with provisionalcompleter('ignore'):
            list(ip.Completer.completions('idx.', 4))