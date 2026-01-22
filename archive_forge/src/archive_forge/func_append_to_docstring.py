import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def append_to_docstring(message: str) -> Callable[[Fn], Fn]:
    """
    Create a decorator which appends passed message to the function's docstring.

    Parameters
    ----------
    message : str
        Message to append.

    Returns
    -------
    callable
    """

    def decorator(func: Fn) -> Fn:
        to_append = align_indents(func.__doc__ or '', message)
        return Appender(to_append)(func)
    return decorator