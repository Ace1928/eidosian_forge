import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class _Filter:
    """
    Given a list of patterns, create a callable that will be true only if
    the input matches at least one of the patterns.
    """

    def __init__(self, *patterns: str):
        self._patterns = dict.fromkeys(patterns)

    def __call__(self, item: str) -> bool:
        return any((fnmatchcase(item, pat) for pat in self._patterns))

    def __contains__(self, item: str) -> bool:
        return item in self._patterns