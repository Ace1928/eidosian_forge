from __future__ import annotations
import functools
from typing import Any, Callable, cast
class MarshmallowHook:
    __marshmallow_hook__: dict[tuple[str, bool] | str, Any] | None = None