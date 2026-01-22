from __future__ import annotations
import sys
import types
from typing import (
class ASGIVersions(TypedDict):
    spec_version: str
    version: Literal['2.0'] | Literal['3.0']