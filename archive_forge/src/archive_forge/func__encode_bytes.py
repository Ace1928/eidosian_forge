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
def _encode_bytes(self, obj: bytes | memoryview) -> BytesRep:
    buffer = Buffer(make_id(), obj)
    data: Buffer | str
    if self._deferred:
        self._buffers.append(buffer)
        data = buffer
    else:
        data = buffer.to_base64()
    return BytesRep(type='bytes', data=data)