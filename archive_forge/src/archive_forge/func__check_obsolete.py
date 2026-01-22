from __future__ import annotations
import warnings
from typing import Any
def _check_obsolete(key: str) -> None:
    if key in _obsolete_constants and key not in _aliases:
        warnings.warn(f"Constant '{key}' is not in current {_current_codata} data set", ConstantWarning, stacklevel=3)