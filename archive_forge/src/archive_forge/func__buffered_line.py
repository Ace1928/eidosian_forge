from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _buffered_line(self) -> list[Scalar]:
    """
        Return a line from buffer, filling buffer if required.
        """
    if len(self.buf) > 0:
        return self.buf[0]
    else:
        return self._next_line()