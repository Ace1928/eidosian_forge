from __future__ import annotations
import collections.abc
import datetime as dt
import math
import typing
from enum import Enum
from functools import total_ordering
from typing import Any
from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma
def _carry(value1: float, value2: float, ratio: float, unit: Unit, min_unit: Unit, suppress: typing.Iterable[Unit]) -> tuple[float, float]:
    """Return a tuple with two values.

    If the unit is in `suppress`, multiply `value1` by `ratio` and add it to `value2`
    (carry to right). The idea is that if we cannot represent `value1` we need to
    represent it in a lower unit.

    >>> from humanize.time import _carry, Unit
    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [Unit.DAYS])
    (0, 54)

    If the unit is the minimum unit, `value2` is divided by `ratio` and added to
    `value1` (carry to left). We assume that `value2` has a lower unit so we need to
    carry it to `value1`.

    >>> _carry(2, 6, 24, Unit.DAYS, Unit.DAYS, [])
    (2.25, 0)

    Otherwise, just return the same input:

    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [])
    (2, 6)
    """
    if unit == min_unit:
        return (value1 + value2 / ratio, 0)
    if unit in suppress:
        return (0, value2 + value1 * ratio)
    return (value1, value2)