from __future__ import annotations
import re
import sys
from typing import (
from trio._util import final
def _unroll_exceptions(self, exceptions: Iterable[BaseException]) -> Iterable[BaseException]:
    """Used in non-strict mode."""
    res: list[BaseException] = []
    for exc in exceptions:
        if isinstance(exc, BaseExceptionGroup):
            res.extend(self._unroll_exceptions(exc.exceptions))
        else:
            res.append(exc)
    return res