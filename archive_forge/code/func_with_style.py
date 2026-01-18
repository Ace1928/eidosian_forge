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
@contextmanager
def with_style(self, x):

    class IdentityClass:

        def __getattr__(self, name):
            return lambda y: y
    yield IdentityClass()