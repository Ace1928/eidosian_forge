from __future__ import annotations
from typing import (
@classmethod
def _instancecheck(cls, inst) -> bool:
    return _check(inst) and (not isinstance(inst, type))