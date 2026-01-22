from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
def _string_length_check(text):
    if MAX_STRING_LENGTH is not None and len(text) > MAX_STRING_LENGTH:
        msg = 'too many characters in string'
        raise ValueError(msg)