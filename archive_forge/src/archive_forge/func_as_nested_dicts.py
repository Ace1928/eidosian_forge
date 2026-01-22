from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Mapping, Tuple
from ufoLib2.serde import serde
def as_nested_dicts(self) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for (left, right), value in self.items():
        result.setdefault(left, {})[right] = value
    return result