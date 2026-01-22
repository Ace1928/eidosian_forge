from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _describe_option(pat: str='', _print_desc: bool=True) -> str | None:
    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError('No such keys(s)')
    s = '\n'.join([_build_option_description(k) for k in keys])
    if _print_desc:
        print(s)
        return None
    return s