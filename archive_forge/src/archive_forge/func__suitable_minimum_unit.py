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
def _suitable_minimum_unit(min_unit: Unit, suppress: typing.Iterable[Unit]) -> Unit:
    """Return a minimum unit suitable that is not suppressed.

    If not suppressed, return the same unit:

    >>> from humanize.time import _suitable_minimum_unit, Unit
    >>> _suitable_minimum_unit(Unit.HOURS, []).name
    'HOURS'

    But if suppressed, find a unit greater than the original one that is not
    suppressed:

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]).name
    'DAYS'

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS]).name
    'MONTHS'
    """
    if min_unit in suppress:
        for unit in Unit:
            if unit > min_unit and unit not in suppress:
                return unit
        msg = 'Minimum unit is suppressed and no suitable replacement was found'
        raise ValueError(msg)
    return min_unit