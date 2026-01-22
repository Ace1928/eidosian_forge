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
def _wrap_decimal_thousands(formatter: Callable, decimal: str, thousands: str | None) -> Callable:
    """
    Takes a string formatting function and wraps logic to deal with thousands and
    decimal parameters, in the case that they are non-standard and that the input
    is a (float, complex, int).
    """

    def wrapper(x):
        if is_float(x) or is_integer(x) or is_complex(x):
            if decimal != '.' and thousands is not None and (thousands != ','):
                return formatter(x).replace(',', '§_§-').replace('.', decimal).replace('§_§-', thousands)
            elif decimal != '.' and (thousands is None or thousands == ','):
                return formatter(x).replace('.', decimal)
            elif decimal == '.' and thousands is not None and (thousands != ','):
                return formatter(x).replace(',', thousands)
        return formatter(x)
    return wrapper