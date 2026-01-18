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
def write_multi_index(self, key: str, index: MultiIndex) -> None:
    setattr(self.attrs, f'{key}_nlevels', index.nlevels)
    for i, (lev, level_codes, name) in enumerate(zip(index.levels, index.codes, index.names)):
        if isinstance(lev.dtype, ExtensionDtype):
            raise NotImplementedError('Saving a MultiIndex with an extension dtype is not supported.')
        level_key = f'{key}_level{i}'
        conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
        self.write_array(level_key, conv_level.values)
        node = getattr(self.group, level_key)
        node._v_attrs.kind = conv_level.kind
        node._v_attrs.name = name
        setattr(node._v_attrs, f'{key}_name{name}', name)
        label_key = f'{key}_label{i}'
        self.write_array(label_key, level_codes)