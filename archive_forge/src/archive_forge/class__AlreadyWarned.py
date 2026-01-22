from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import (
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
class _AlreadyWarned:

    def __init__(self):
        self.warned_already = False