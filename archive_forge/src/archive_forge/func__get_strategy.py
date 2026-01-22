import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
def _get_strategy() -> str:
    return strategy