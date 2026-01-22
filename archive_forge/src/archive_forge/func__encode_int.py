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
def _encode_int(self, obj: int) -> AnyRep:
    if -_MAX_SAFE_INT < obj <= _MAX_SAFE_INT:
        return obj
    else:
        warn('out of range integer may result in loss of precision', BokehUserWarning)
        return self._encode_float(float(obj))