import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def _mask_positional_args(name: str) -> List[Optional[str]]:
    """Create a section name representation that masks names
    of positional arguments to retain their order in sorts."""
    stable_name = cast(List[Optional[str]], name.split('.'))
    for i in range(1, len(stable_name)):
        if stable_name[i - 1] == '*':
            stable_name[i] = None
    return stable_name