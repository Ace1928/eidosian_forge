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
class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None:
        self.colspecs = kwds.pop('colspecs')
        self.infer_nrows = kwds.pop('infer_nrows')
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader:
        return FixedWidthReader(f, self.colspecs, self.delimiter, self.comment, self.skiprows, self.infer_nrows)

    def _remove_empty_lines(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [line for line in lines if any((not isinstance(e, str) or e.strip() for e in line))]