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
def _escape_latex_math(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which either are surrounded
    by two characters ``$`` or start with the character ``\\(`` and end with ``\\)``,
    are preserved without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    ps_d = re.compile('\\$.*?\\$').search(s, 0)
    ps_p = re.compile('\\(.*?\\)').search(s, 0)
    mode = []
    if ps_d:
        mode.append(ps_d.span()[0])
    if ps_p:
        mode.append(ps_p.span()[0])
    if len(mode) == 0:
        return _escape_latex(s.replace('rt8§=§7wz', '\\$'))
    if s[mode[0]] == '$':
        return _math_mode_with_dollar(s.replace('rt8§=§7wz', '\\$'))
    if s[mode[0] - 1:mode[0] + 1] == '\\(':
        return _math_mode_with_parentheses(s.replace('rt8§=§7wz', '\\$'))
    else:
        return _escape_latex(s.replace('rt8§=§7wz', '\\$'))