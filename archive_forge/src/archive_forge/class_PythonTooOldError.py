from __future__ import annotations
from typing import ClassVar
class PythonTooOldError(RuntimeError):
    """
    It was attempted to use an *attrs* feature that requires a newer Python
    version.

    .. versionadded:: 18.2.0
    """