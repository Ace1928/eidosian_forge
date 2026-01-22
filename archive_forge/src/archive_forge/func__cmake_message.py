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
def _cmake_message(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    if len(args) < 1:
        return self._gen_exception('message', 'takes at least 1 argument', tline)
    if args[0].upper().strip() not in ['FATAL_ERROR', 'SEND_ERROR']:
        return
    self.errors += [' '.join(args[1:])]