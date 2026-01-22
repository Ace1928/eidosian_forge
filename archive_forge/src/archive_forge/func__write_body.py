from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def _write_body(self, indent: int) -> None:
    self.write('<tbody>', indent)
    fmt_values = self._get_formatted_values()
    if self.fmt.index and isinstance(self.frame.index, MultiIndex):
        self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
    else:
        self._write_regular_rows(fmt_values, indent + self.indent_delta)
    self.write('</tbody>', indent)