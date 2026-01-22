from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
def _format_attrs(self) -> list[tuple[str_t, str_t | int | bool | None]]:
    """
        Return a list of tuples of the (attr,formatted_value).
        """
    attrs: list[tuple[str_t, str_t | int | bool | None]] = []
    if not self._is_multi:
        attrs.append(('dtype', f"'{self.dtype}'"))
    if self.name is not None:
        attrs.append(('name', default_pprint(self.name)))
    elif self._is_multi and any((x is not None for x in self.names)):
        attrs.append(('names', default_pprint(self.names)))
    max_seq_items = get_option('display.max_seq_items') or len(self)
    if len(self) > max_seq_items:
        attrs.append(('length', len(self)))
    return attrs