from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@pandas.util.cache_readonly
def _Series(self):
    from .series import Series
    return Series