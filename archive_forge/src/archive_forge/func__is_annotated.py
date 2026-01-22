from typing import Optional
import inspect
import sys
import warnings
from functools import wraps
def _is_annotated(obj) -> bool:
    return hasattr(obj, '_annotated') and obj._annotated == obj.__name__