from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def add_separator_line(self) -> None:
    separator_line = self.SPACING.join([_put_str('-' * header_colwidth, gross_colwidth) for header_colwidth, gross_colwidth in zip(self.header_column_widths, self.gross_column_widths)])
    self._lines.append(separator_line)