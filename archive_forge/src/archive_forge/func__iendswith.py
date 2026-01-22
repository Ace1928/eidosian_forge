from __future__ import annotations
import os
import pathlib
import typing as ty
def _iendswith(whole: str, end: str) -> bool:
    return whole.lower().endswith(end.lower())