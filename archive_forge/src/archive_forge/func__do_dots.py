from __future__ import annotations
import re
from typing import (
def _do_dots(self, value: Any, *dots: str) -> Any:
    """Evaluate dotted expressions at run-time."""
    for dot in dots:
        try:
            value = getattr(value, dot)
        except AttributeError:
            try:
                value = value[dot]
            except (TypeError, KeyError) as exc:
                raise TempliteValueError(f"Couldn't evaluate {value!r}.{dot}") from exc
        if callable(value):
            value = value()
    return value