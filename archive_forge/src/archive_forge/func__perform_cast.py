from __future__ import annotations
import os
import typing
from pathlib import Path
def _perform_cast(self, key: str, value: typing.Any, cast: typing.Callable[[typing.Any], typing.Any] | None=None) -> typing.Any:
    if cast is None or value is None:
        return value
    elif cast is bool and isinstance(value, str):
        mapping = {'true': True, '1': True, 'false': False, '0': False}
        value = value.lower()
        if value not in mapping:
            raise ValueError(f"Config '{key}' has value '{value}'. Not a valid bool.")
        return mapping[value]
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"Config '{key}' has value '{value}'. Not a valid {cast.__name__}.")