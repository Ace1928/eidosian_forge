import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def _error(self, msg: str, *args: Any, _level_str: str=None, **kwargs: Any):
    """Prints a formatted error message.

        For arguments, see `_format_msg`.
        """
    if _level_str is None:
        raise ValueError('Log level not set.')
    self.print(cf.red(msg), *args, _level_str=_level_str, **kwargs)