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
def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
    """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
    try:
        assert self.data is not None
        line = next(self.data)
        assert isinstance(line, list)
        return line
    except csv.Error as e:
        if self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
            msg = str(e)
            if 'NULL byte' in msg or 'line contains NUL' in msg:
                msg = "NULL byte detected. This byte cannot be processed in Python's native csv library at the moment, so please pass in engine='c' instead"
            if self.skipfooter > 0:
                reason = "Error could possibly be due to parsing errors in the skipped footer rows (the skipfooter keyword is only applied after Python's csv library has parsed all rows)."
                msg += '. ' + reason
            self._alert_malformed(msg, row_num)
        return None