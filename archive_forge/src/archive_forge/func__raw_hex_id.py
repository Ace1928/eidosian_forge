from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError
def _raw_hex_id(obj) -> str:
    """Return the padded hexadecimal id of ``obj``."""
    packed = struct.pack('@P', id(obj))
    return ''.join([_replacer(x) for x in packed])