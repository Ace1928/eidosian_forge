from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
def _get_repr_footer(self) -> str:
    """
        Returns a string representation of the footer.
        """
    category_strs = self._repr_categories()
    dtype = str(self.categories.dtype)
    levheader = f'Categories ({len(self.categories)}, {dtype}): '
    width, _ = get_terminal_size()
    max_width = get_option('display.width') or width
    if console.in_ipython_frontend():
        max_width = 0
    levstring = ''
    start = True
    cur_col_len = len(levheader)
    sep_len, sep = (3, ' < ') if self.ordered else (2, ', ')
    linesep = f'{sep.rstrip()}\n'
    for val in category_strs:
        if max_width != 0 and cur_col_len + sep_len + len(val) > max_width:
            levstring += linesep + ' ' * (len(levheader) + 1)
            cur_col_len = len(levheader) + 1
        elif not start:
            levstring += sep
            cur_col_len += len(val)
        levstring += val
        start = False
    return f'{levheader}[{levstring.replace(' < ... < ', ' ... ')}]'