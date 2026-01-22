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
def _parse_latex_options_strip(value: str | float, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
    return str(value).replace(arg, '').replace('/*', '').replace('*/', '').strip()