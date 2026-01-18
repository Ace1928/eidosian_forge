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
def validate_attr(self, append) -> None:
    """validate that we have the same order as the existing & same dtype"""
    if append:
        existing_fields = getattr(self.attrs, self.kind_attr, None)
        if existing_fields is not None and existing_fields != list(self.values):
            raise ValueError('appended items do not match existing items in table!')
        existing_dtype = getattr(self.attrs, self.dtype_attr, None)
        if existing_dtype is not None and existing_dtype != self.dtype:
            raise ValueError('appended items dtype do not match existing items dtype in table!')