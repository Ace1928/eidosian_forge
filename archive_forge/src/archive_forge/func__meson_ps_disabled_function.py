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
def _meson_ps_disabled_function(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    if not args:
        mlog.error('Invalid preload.cmake script! At least one argument to `meson_ps_disabled_function` is expected')
        return
    mlog.warning(f'The CMake function "{args[0]}" was disabled to avoid compatibility issues with Meson.')