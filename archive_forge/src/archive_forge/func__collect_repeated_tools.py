from __future__ import annotations
import logging # isort:skip
import itertools
import re
from dataclasses import dataclass
from typing import (
from ..models import (
from ..models.tools import (
from ..util.warnings import warn
def _collect_repeated_tools(tool_objs: list[Tool]) -> Iterator[Tool]:

    @dataclass(frozen=True)
    class Item:
        obj: Tool
        properties: dict[str, Any]
    key: Callable[[Tool], str] = lambda obj: obj.__class__.__name__
    for _, group in itertools.groupby(sorted(tool_objs, key=key), key=key):
        rest = [Item(obj, obj.properties_with_values()) for obj in group]
        while len(rest) > 1:
            head, *rest = rest
            for item in rest:
                if item.properties == head.properties:
                    yield item.obj