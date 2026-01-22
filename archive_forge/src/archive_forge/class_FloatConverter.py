from __future__ import annotations
import re
import typing as t
import uuid
from urllib.parse import quote
class FloatConverter(NumberConverter):
    """This converter only accepts floating point values::

        Rule("/probability/<float:probability>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/offset/<float(signed=True):offset>")

    :param map: The :class:`Map`.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """
    regex = '\\d+\\.\\d+'
    num_convert = float

    def __init__(self, map: Map, min: float | None=None, max: float | None=None, signed: bool=False) -> None:
        super().__init__(map, min=min, max=max, signed=signed)