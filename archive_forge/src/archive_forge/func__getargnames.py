import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def _getargnames(self, func: Callable) -> List[str]:
    try:
        return getfullargspec(func).args
    except TypeError:
        if hasattr(func, 'func_code'):
            code = func.func_code
            return code.co_varnames[:code.co_argcount]
        raise