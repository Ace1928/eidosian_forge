from __future__ import annotations
from abc import (
from typing import (
from pandas.compat import (
from pandas.core.dtypes.common import is_list_like
def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
    return pa.types.is_struct(pyarrow_dtype)