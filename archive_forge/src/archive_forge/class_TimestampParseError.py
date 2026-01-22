from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
class TimestampParseError(ValueError):
    """
    Raised when an SRT timestamp could not be parsed.
    """