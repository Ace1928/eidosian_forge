from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _check_write_command_response(result: Mapping[str, Any]) -> None:
    """Backward compatibility helper for write command error handling."""
    write_errors = result.get('writeErrors')
    if write_errors:
        _raise_last_write_error(write_errors)
    wce = _get_wce_doc(result)
    if wce:
        _raise_write_concern_error(wce)