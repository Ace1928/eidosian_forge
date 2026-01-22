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
def _search_replace_num_columns(self, lines: list[list[Scalar]], search: str, replace: str) -> list[list[Scalar]]:
    ret = []
    for line in lines:
        rl = []
        for i, x in enumerate(line):
            if not isinstance(x, str) or search not in x or i in self._no_thousands_columns or (not self.num.search(x.strip())):
                rl.append(x)
            else:
                rl.append(x.replace(search, replace))
        ret.append(rl)
    return ret