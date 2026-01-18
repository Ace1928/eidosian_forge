from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_signature(self):
    sig = inspect.signature(DataFrame.reindex)
    parameters = set(sig.parameters)
    assert parameters == {'self', 'labels', 'index', 'columns', 'axis', 'limit', 'copy', 'level', 'method', 'fill_value', 'tolerance'}