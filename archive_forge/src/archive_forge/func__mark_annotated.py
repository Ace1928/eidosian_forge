from typing import Optional
import inspect
import sys
import warnings
from functools import wraps
def _mark_annotated(obj) -> None:
    if hasattr(obj, '__name__'):
        obj._annotated = obj.__name__