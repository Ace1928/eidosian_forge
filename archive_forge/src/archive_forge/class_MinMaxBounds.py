from __future__ import annotations
import logging # isort:skip
import base64
import datetime  # lgtm [py/import-and-import-from]
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from ...util.serialization import convert_datetime_type
from .. import enums
from .auto import Auto
from .bases import Property
from .container import Seq, Tuple
from .datetime import Datetime, TimeDelta
from .either import Either
from .enum import Enum
from .nullable import Nullable
from .numeric import Float, Int
from .primitive import String
from .string import Regex
class MinMaxBounds(Either):
    """ Accept (min, max) bounds tuples for use with Ranges.

    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether your range is
    increasing or decreasing, the first item should be the minimum value of the range and the
    second item should be the maximum. Setting min > max will result in a ``ValueError``.

    Setting bounds to None will allow your plot to pan/zoom as far as you want. If you only
    want to constrain one end of the plot, you can set min or max to
    ``None`` e.g. ``DataRange1d(bounds=(None, 12))`` """

    def __init__(self, default='auto', *, accept_datetime: bool=False, help: str | None=None) -> None:
        types = (Auto, Tuple(Float, Float), Tuple(Nullable(Float), Float), Tuple(Float, Nullable(Float)), Tuple(TimeDelta, TimeDelta), Tuple(Nullable(TimeDelta), TimeDelta), Tuple(TimeDelta, Nullable(TimeDelta)))
        if accept_datetime:
            types = (*types, Tuple(Datetime, Datetime), Tuple(Nullable(Datetime), Datetime), Tuple(Datetime, Nullable(Datetime)))
        super().__init__(*types, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if value[0] is None or value[1] is None:
            return
        value = list(value)
        if isinstance(value[0], datetime.datetime):
            value[0] = convert_datetime_type(value[0])
        if isinstance(value[1], datetime.datetime):
            value[1] = convert_datetime_type(value[1])
        if value[0] < value[1]:
            return
        msg = '' if not detail else 'Invalid bounds: maximum smaller than minimum. Correct usage: bounds=(min, max)'
        raise ValueError(msg)