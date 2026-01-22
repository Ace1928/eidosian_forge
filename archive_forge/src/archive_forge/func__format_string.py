import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _format_string(value: bytes) -> bytes:
    if value.startswith((b' ', b'\t')) or value.endswith((b' ', b'\t')) or b'#' in value:
        return b'"' + _escape_value(value) + b'"'
    else:
        return _escape_value(value)