from __future__ import annotations
from abc import (
from typing import (
from pandas.compat import (
from pandas.core.dtypes.common import is_list_like
@property
def _pa_array(self):
    return self._data.array._pa_array