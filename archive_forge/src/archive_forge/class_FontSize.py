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
class FontSize(String):
    _font_size_re = re.compile('^[0-9]+(.[0-9]+)?(%|em|ex|ch|ic|rem|vw|vh|vi|vb|vmin|vmax|cm|mm|q|in|pc|pt|px)$', re.I)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if isinstance(value, str):
            if len(value) == 0:
                msg = '' if not detail else 'empty string is not a valid font size value'
                raise ValueError(msg)
            elif not self._font_size_re.match(value):
                msg = '' if not detail else f'{value!r} is not a valid font size value'
                raise ValueError(msg)