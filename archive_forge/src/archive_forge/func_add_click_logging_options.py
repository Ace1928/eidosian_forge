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
def add_click_logging_options(f: Callable) -> Callable:
    for option in reversed(CLICK_LOGGING_OPTIONS):
        f = option(f)

    @wraps(f)
    def wrapper(*args, log_style=None, log_color=None, verbose=None, **kwargs):
        cli_logger.configure(log_style, log_color, verbose)
        return f(*args, **kwargs)
    return wrapper