import collections
import contextlib
import itertools
import pathlib
import operator
import re
import warnings
from . import abc
from ._itertools import only
from .compat.py39 import ZipPath
@classmethod
def _candidate_paths(cls, path_str):
    yield pathlib.Path(path_str)
    yield from cls._resolve_zip_path(path_str)