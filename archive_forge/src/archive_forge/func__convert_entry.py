from __future__ import division
import sys as _sys
import datetime as _datetime
import uuid as _uuid
import traceback as _traceback
import os as _os
import logging as _logging
from syslog import (LOG_EMERG, LOG_ALERT, LOG_CRIT, LOG_ERR,
from ._journal import __version__, sendv, stream_fd
from ._reader import (_Reader, NOP, APPEND, INVALIDATE,
from . import id128 as _id128
def _convert_entry(self, entry):
    """Convert entire journal entry utilising _convert_field."""
    result = {}
    for key, value in entry.items():
        if isinstance(value, list):
            result[key] = [self._convert_field(key, val) for val in value]
        else:
            result[key] = self._convert_field(key, value)
    return result