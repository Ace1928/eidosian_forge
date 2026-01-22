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
class HatchPatternType(Either):
    """ Accept built-in fill hatching specifications.

    Accepts either "long" names, e.g. "horizontal-wave" or the single letter
    abbreviations, e.g. "v"

    """

    def __init__(self, default=[], *, help: str | None=None) -> None:
        types = (Enum(enums.HatchPattern), Enum(enums.HatchPatternAbbreviation), String)
        super().__init__(*types, default=default, help=help)

    def __str__(self) -> str:
        return self.__class__.__name__