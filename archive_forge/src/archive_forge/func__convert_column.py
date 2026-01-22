from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
def _convert_column(self, values: np.ndarray, old_values: np.ndarray | 'pd.Series') -> np.ndarray | List:
    dtype = old_values.dtype
    converted: List | np.ndarray | None = None
    if dtype.kind == 'M':
        if values.dtype.kind in 'if':
            NATs = np.isnan(values)
            converted = np.where(NATs, np.nan, values * 1000000.0).astype(dtype)
    elif dtype.kind == 'O':
        if all((isinstance(ov, dt.date) for ov in old_values)) and (not all((isinstance(iv, dt.date) for iv in values))):
            new_values = []
            for iv in values:
                if isinstance(iv, dt.datetime):
                    iv = iv.date()
                elif not isinstance(iv, dt.date):
                    iv = dt.date.fromtimestamp(iv / 1000)
                new_values.append(iv)
            converted = new_values
    elif 'pandas' in sys.modules:
        import pandas as pd
        if Version(pd.__version__) >= Version('1.1.0'):
            from pandas.core.arrays.masked import BaseMaskedDtype
            if isinstance(dtype, BaseMaskedDtype):
                values = [dtype.na_value if v == '<NA>' else v for v in values]
        converted = pd.Series(values).astype(dtype).values
    else:
        converted = values.astype(dtype)
    return values if converted is None else converted