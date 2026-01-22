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
class _ColorfulMock:

    def __init__(self):
        self.identity = lambda x: x
        self.colorful = self
        self.colormode = None
        self.NO_COLORS = None
        self.ANSI_8_COLORS = None

    def disable(self):
        pass

    @contextmanager
    def with_style(self, x):

        class IdentityClass:

            def __getattr__(self, name):
                return lambda y: y
        yield IdentityClass()

    def __getattr__(self, name):
        if name == 'with_style':
            return self.with_style
        return self.identity