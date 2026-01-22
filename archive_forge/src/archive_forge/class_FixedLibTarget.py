from __future__ import annotations
import typing as T
from typing_extensions import Literal, TypedDict, Required
class FixedLibTarget(_BaseFixedBuildTarget, total=False):
    name: Required[str]
    proc_macro: bool