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
def with_csv_dialect(name: str, **kwargs) -> Generator[None, None, None]:
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    import csv
    _BUILTIN_DIALECTS = {'excel', 'excel-tab', 'unix'}
    if name in _BUILTIN_DIALECTS:
        raise ValueError('Cannot override builtin dialect.')
    csv.register_dialect(name, **kwargs)
    try:
        yield
    finally:
        csv.unregister_dialect(name)