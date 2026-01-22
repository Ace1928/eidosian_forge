from __future__ import annotations
from typing import ClassVar
class NotAnAttrsClassError(ValueError):
    """
    A non-*attrs* class has been passed into an *attrs* function.

    .. versionadded:: 16.2.0
    """