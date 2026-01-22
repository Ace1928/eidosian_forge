from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _str_escape(x, escape):
    """if escaping: only use on str, else return input"""
    if isinstance(x, str):
        if escape == 'html':
            return escape_html(x)
        elif escape == 'latex':
            return _escape_latex(x)
        elif escape == 'latex-math':
            return _escape_latex_math(x)
        else:
            raise ValueError(f"`escape` only permitted in {{'html', 'latex', 'latex-math'}}, got {escape}")
    return x