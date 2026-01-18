from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
def unpack_1tuple(tup):
    """
    If we have a length-1 tuple/list that contains a slice, unpack to just
    the slice.

    Notes
    -----
    The list case is deprecated.
    """
    if len(tup) == 1 and isinstance(tup[0], slice):
        if isinstance(tup, list):
            raise ValueError('Indexing with a single-item list containing a slice is not allowed. Pass a tuple instead.')
        return tup[0]
    return tup