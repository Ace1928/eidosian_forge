from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
import uuid
from pandas._config import using_copy_on_write
from pandas.compat import PYPY
from pandas.errors import ChainedAssignmentError
from pandas import set_option
from pandas.io.common import get_handle
@contextmanager
def use_numexpr(use, min_elements=None) -> Generator[None, None, None]:
    from pandas.core.computation import expressions as expr
    if min_elements is None:
        min_elements = expr._MIN_ELEMENTS
    olduse = expr.USE_NUMEXPR
    oldmin = expr._MIN_ELEMENTS
    set_option('compute.use_numexpr', use)
    expr._MIN_ELEMENTS = min_elements
    try:
        yield
    finally:
        expr._MIN_ELEMENTS = oldmin
        set_option('compute.use_numexpr', olduse)