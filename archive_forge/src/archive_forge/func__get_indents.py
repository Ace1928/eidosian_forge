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
def _get_indents(source: Union[list, str]) -> list:
    """
    Compute indentation for each line of the source string.

    Parameters
    ----------
    source : str or list of str
        String to compute indents for. Passed list considered
        as a list of lines of the source string.

    Returns
    -------
    list of ints
        List containing computed indents for each line.
    """
    indents = []
    if not isinstance(source, list):
        source = source.splitlines()
    for line in source:
        if not line.strip():
            continue
        for pos, ch in enumerate(line):
            if ch != ' ':
                break
        indents.append(pos)
    return indents