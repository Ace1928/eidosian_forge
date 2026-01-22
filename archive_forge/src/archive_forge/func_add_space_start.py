from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def add_space_start(x: Any) -> Any:
    x['completion'] = x['completion'].apply(lambda s: ('' if s.startswith(' ') else ' ') + s)
    return x