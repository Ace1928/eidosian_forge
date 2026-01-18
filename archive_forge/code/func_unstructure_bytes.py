from __future__ import annotations
import sys
from functools import partial
from typing import Any, Callable, Tuple, Type, cast
from attrs import fields, has, resolve_types
from cattrs import Converter
from cattrs.gen import (
from fontTools.misc.transform import Transform
def unstructure_bytes(v: bytes) -> str:
    return (b64encode(v) if v else b'').decode('utf8')