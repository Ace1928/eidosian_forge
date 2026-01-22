import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def _check_roundtrip_table(obj, comparator, path, compression=False):
    options = {}
    if compression:
        options['complib'] = 'blosc'
    with ensure_clean_store(path, 'w', **options) as store:
        store.put('obj', obj, format='table')
        retrieved = store['obj']
        comparator(retrieved, obj)