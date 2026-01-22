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
def _repr_categories(self) -> list[str]:
    """
        return the base repr for the categories
        """
    max_categories = 10 if get_option('display.max_categories') == 0 else get_option('display.max_categories')
    from pandas.io.formats import format as fmt
    format_array = partial(fmt.format_array, formatter=None, quoting=QUOTE_NONNUMERIC)
    if len(self.categories) > max_categories:
        num = max_categories // 2
        head = format_array(self.categories[:num]._values)
        tail = format_array(self.categories[-num:]._values)
        category_strs = head + ['...'] + tail
    else:
        category_strs = format_array(self.categories._values)
    category_strs = [x.strip() for x in category_strs]
    return category_strs