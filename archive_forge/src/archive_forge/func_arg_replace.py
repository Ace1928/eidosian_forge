from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
def arg_replace(match: T.Match[str]) -> str:
    idx = int(match.group(1))
    if idx >= len(arg_strings):
        raise InvalidArguments(f'Format placeholder @{idx}@ out of range.')
    return arg_strings[idx]