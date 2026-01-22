from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _select_options(pat: str) -> list[str]:
    """
    returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """
    if pat in _registered_options:
        return [pat]
    keys = sorted(_registered_options.keys())
    if pat == 'all':
        return keys
    return [k for k in keys if re.search(pat, k, re.I)]