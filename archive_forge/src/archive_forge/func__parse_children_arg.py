from __future__ import annotations
import logging # isort:skip
import math
from collections import defaultdict
from typing import (
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import (
from .util.dataclasses import dataclass
from .util.warnings import warn
def _parse_children_arg(*args: L | list[L], children: list[L] | None=None) -> list[L]:
    if len(args) > 0 and children is not None:
        raise ValueError("'children' keyword cannot be used with positional arguments")
    if not children:
        if len(args) == 1:
            [arg] = args
            if isinstance(arg, list):
                return arg
        return list(args)
    return children