import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def _get_emulated_is_finalizing() -> Callable[[], bool]:
    L = []
    atexit.register(lambda: L.append(None))

    def is_finalizing() -> bool:
        return L != []
    return is_finalizing