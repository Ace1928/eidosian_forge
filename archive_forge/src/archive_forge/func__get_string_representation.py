from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
def _get_string_representation(self) -> str:
    if self.fmt.frame.empty:
        return self._empty_info_line
    strcols = self._get_strcols()
    if self.line_width is None:
        return self.adj.adjoin(1, *strcols)
    if self._need_to_wrap_around:
        return self._join_multiline(strcols)
    return self._fit_strcols_to_terminal_width(strcols)