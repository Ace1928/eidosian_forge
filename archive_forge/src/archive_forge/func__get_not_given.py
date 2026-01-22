from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
@functools.lru_cache
def _get_not_given() -> Optional[Type]:
    try:
        from openai._types import NotGiven
        return NotGiven
    except ImportError:
        return None