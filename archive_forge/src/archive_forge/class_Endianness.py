from __future__ import annotations
import typing
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.dtypes import (
class Endianness:
    """Enum indicating the byte-order of a data-type."""
    LITTLE = '<'
    BIG = '>'
    NATIVE = '='
    NA = '|'