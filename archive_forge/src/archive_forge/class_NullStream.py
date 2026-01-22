from __future__ import annotations
import os
from functools import wraps
from typing import Hashable, TypeVar
class NullStream:
    """A fake stream with a no-op write."""

    def write(self, *args):
        """
        Does nothing...
        """