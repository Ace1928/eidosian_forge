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
def _math_mode_with_parentheses(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``\\(`` and end with ``\\)``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\(', 'LEFT§=§6yzLEFT').replace('\\)', 'RIGHTab5§=§RIGHT')
    res = []
    for item in re.split('LEFT§=§6yz|ab5§=§RIGHT', s):
        if item.startswith('LEFT') and item.endswith('RIGHT'):
            res.append(item.replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        elif 'LEFT' in item and 'RIGHT' in item:
            res.append(_escape_latex(item).replace('LEFT', '\\(').replace('RIGHT', '\\)'))
        else:
            res.append(_escape_latex(item).replace('LEFT', '\\textbackslash (').replace('RIGHT', '\\textbackslash )'))
    return ''.join(res)