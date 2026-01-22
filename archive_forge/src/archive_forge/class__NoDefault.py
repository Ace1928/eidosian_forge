from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version
from xarray.coding import cftime_offsets
class _NoDefault(Enum):
    """Used by pandas to specify a default value for a deprecated argument.
    Copied from pandas._libs.lib._NoDefault.

    See also:
    - pandas-dev/pandas#30788
    - pandas-dev/pandas#40684
    - pandas-dev/pandas#40715
    - pandas-dev/pandas#47045
    """
    no_default = 'NO_DEFAULT'

    def __repr__(self) -> str:
        return '<no_default>'