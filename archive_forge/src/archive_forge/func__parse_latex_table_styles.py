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
def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    """
    Return the first 'props' 'value' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [{'selector': 'foo', 'props': [('attr','value')]},
    ...                 {'selector': 'bar', 'props': [('attr', 'overwritten')]},
    ...                 {'selector': 'bar', 'props': [('a1', 'baz'), ('a2', 'ignore')]}]
    >>> _parse_latex_table_styles(table_styles, selector='bar')
    'baz'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    """
    for style in table_styles[::-1]:
        if style['selector'] == selector:
            return str(style['props'][0][1]).replace('§', ':')
    return None