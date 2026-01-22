from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
def _rich_tag_if_enabled(x: str, tag: str) -> str:
    x = rich.markup.escape(_strings.strip_ansi_sequences(x))
    return x if not USE_RICH else f'[{tag}]{x}[/{tag}]'