from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def _apply_index(self, func: Callable, axis: Axis=0, level: Level | list[Level] | None=None, method: str='apply', **kwargs) -> Styler:
    axis = self.data._get_axis_number(axis)
    obj = self.index if axis == 0 else self.columns
    levels_ = refactor_levels(level, obj)
    data = DataFrame(obj.to_list()).loc[:, levels_]
    if method == 'apply':
        result = data.apply(func, axis=0, **kwargs)
    elif method == 'map':
        result = data.map(func, **kwargs)
    self._update_ctx_header(result, axis)
    return self