import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
def tokenize_version(version_string: str):
    """Parse version string into a tuple for version comparisons.

    Usage: tokenize_version('3.8') < tokenize_version('3.8.1').
    """
    tokens = []
    for component in version_string.split('.'):
        match = re.match('(\\d*)(.*)', component)
        assert match is not None, f'Cannot parse component {component}'
        number_str, tail = match.group(1, 2)
        try:
            number = int(number_str)
        except ValueError:
            number = -1
        tokens += [number, tail]
    return tuple(tokens)