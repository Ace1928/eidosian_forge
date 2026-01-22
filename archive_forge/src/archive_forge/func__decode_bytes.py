from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
def _decode_bytes(self, obj: BytesRep) -> bytes:
    data = obj['data']
    if isinstance(data, str):
        return base64.b64decode(data)
    elif isinstance(data, Buffer):
        buffer = data
    else:
        id = data['id']
        if id in self._buffers:
            buffer = self._buffers[id]
        else:
            self.error(f"can't resolve buffer '{id}'")
    return buffer.data