from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
@property
def _adjusted_tr_col_num(self) -> int:
    return self.fmt.tr_col_num + 1 if self.fmt.index else self.fmt.tr_col_num