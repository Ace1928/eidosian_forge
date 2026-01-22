import logging
import os
import sys
import warnings
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Generator, Optional, Set, Tuple, Union
import anyio
from ._rust_notify import RustNotify
from .filters import DefaultFilter
def _calc_async_timeout(timeout: Optional[int]) -> int:
    """
    see https://github.com/samuelcolvin/watchfiles/issues/110
    """
    if timeout is None:
        if sys.platform == 'win32':
            return 1000
        else:
            return 5000
    else:
        return timeout