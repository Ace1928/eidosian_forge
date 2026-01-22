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
def _isatty():
    """More robust check for interactive terminal/tty."""
    try:
        return sys.__stdin__.isatty()
    except Exception:
        return False