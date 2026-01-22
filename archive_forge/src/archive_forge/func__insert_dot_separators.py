from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
def _insert_dot_separators(self, strcols: list[list[str]]) -> list[list[str]]:
    str_index = self.fmt._get_formatted_index(self.fmt.tr_frame)
    index_length = len(str_index)
    if self.fmt.is_truncated_horizontally:
        strcols = self._insert_dot_separator_horizontal(strcols, index_length)
    if self.fmt.is_truncated_vertically:
        strcols = self._insert_dot_separator_vertical(strcols, index_length)
    return strcols