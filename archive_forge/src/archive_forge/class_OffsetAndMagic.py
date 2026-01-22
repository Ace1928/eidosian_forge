import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
@dataclasses.dataclass(frozen=True)
class OffsetAndMagic:
    col_offset: int
    magic: str