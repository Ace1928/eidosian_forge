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
def _concatenated_visible_rows(obj, n, row_indices):
    """
            Extract all visible row indices recursively from concatenated stylers.
            """
    row_indices.extend([r + n for r in range(len(obj.index)) if r not in obj.hidden_rows])
    n += len(obj.index)
    for concatenated in obj.concatenated:
        n = _concatenated_visible_rows(concatenated, n, row_indices)
    return n