from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _cmake_add_custom_target(self, tline: CMakeTraceLine) -> None:
    if len(tline.args) < 1:
        return self._gen_exception('add_custom_target', 'requires at least one argument', tline)
    self._cmake_add_custom_command(tline, tline.args[0])