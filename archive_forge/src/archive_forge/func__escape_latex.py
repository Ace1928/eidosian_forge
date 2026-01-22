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
def _escape_latex(s: str) -> str:
    """
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    return s.replace('\\', 'ab2§=§8yz').replace('ab2§=§8yz ', 'ab2§=§8yz\\space ').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('~ ', '~\\space ').replace('~', '\\textasciitilde ').replace('^ ', '^\\space ').replace('^', '\\textasciicircum ').replace('ab2§=§8yz', '\\textbackslash ')