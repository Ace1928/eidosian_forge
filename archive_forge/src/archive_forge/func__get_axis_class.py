from __future__ import annotations
import logging # isort:skip
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from ..core.properties import Datetime
from ..core.property.singletons import Intrinsic
from ..models import (
def _get_axis_class(axis_type: AxisType | None, range_input: Range, dim: Dim) -> tuple[type[Axis] | None, Any]:
    if axis_type is None:
        return (None, {})
    elif axis_type == 'linear':
        return (LinearAxis, {})
    elif axis_type == 'log':
        return (LogAxis, {})
    elif axis_type == 'datetime':
        return (DatetimeAxis, {})
    elif axis_type == 'mercator':
        return (MercatorAxis, dict(dimension='lon' if dim == 0 else 'lat'))
    elif axis_type == 'auto':
        if isinstance(range_input, FactorRange):
            return (CategoricalAxis, {})
        elif isinstance(range_input, Range1d):
            try:
                value = range_input.start
                if Datetime.is_timestamp(value):
                    return (LinearAxis, {})
                Datetime.validate(Datetime(), value)
                return (DatetimeAxis, {})
            except ValueError:
                pass
        return (LinearAxis, {})
    else:
        raise ValueError(f"Unrecognized axis_type: '{axis_type!r}'")