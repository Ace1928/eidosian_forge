from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def update_info(self, info) -> None:
    """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
    for key in self._info_fields:
        value = getattr(self, key, None)
        idx = info.setdefault(self.name, {})
        existing_value = idx.get(key)
        if key in idx and value is not None and (existing_value != value):
            if key in ['freq', 'index_name']:
                ws = attribute_conflict_doc % (key, existing_value, value)
                warnings.warn(ws, AttributeConflictWarning, stacklevel=find_stack_level())
                idx[key] = None
                setattr(self, key, None)
            else:
                raise ValueError(f'invalid info for [{self.name}] for [{key}], existing_value [{existing_value}] conflicts with new value [{value}]')
        elif value is not None or existing_value is not None:
            idx[key] = value