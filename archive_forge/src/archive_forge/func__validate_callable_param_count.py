import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
@classmethod
def _validate_callable_param_count(cls, func: Callable[..., Any], count: int) -> None:
    """Ensure a function has the given number of parameters."""
    signature = inspect.signature(func)
    nparam = len(signature.parameters)
    if nparam != count:
        plural = '' if nparam == 1 else 's'
        raise TypeError(f'{func.__name__} has {nparam} positional argument{plural}, expected {count}')