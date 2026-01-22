import os
import sys
from typing import TYPE_CHECKING, Dict
def _trace_repr(value, size=40):
    value = repr(value)
    if len(value) > size:
        value = value[:size // 2 - 2] + '...' + value[-size // 2 - 1:]
    return value