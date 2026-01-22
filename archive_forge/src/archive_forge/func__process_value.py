from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@staticmethod
def _process_value(value: Any, code: int, dtype: int, offset: int, /) -> Any:
    """Process tag value."""
    if value is None or dtype == 1 or dtype == 7 or (code in TIFF.TAG_READERS) or (not isinstance(value, (bytes, str, tuple))):
        return value
    if dtype == 2:
        try:
            value = bytes2str(stripnull(cast(bytes, value), first=False).strip())
        except UnicodeDecodeError as exc:
            logger().warning(f'<tifffile.TiffTag {code} @{offset}> coercing invalid ASCII to bytes, due to {exc!r}')
        return value
    if code in TIFF.TAG_ENUM:
        t = TIFF.TAG_ENUM[code]
        try:
            value = tuple((t(v) for v in value))
        except ValueError as exc:
            if code not in {259, 317}:
                logger().warning(f'<tifffile.TiffTag {code} @{offset}> raised {exc!r}')
    if len(value) == 1 and code not in TIFF.TAG_TUPLE:
        value = value[0]
    return value