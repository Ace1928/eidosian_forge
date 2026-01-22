from __future__ import annotations
import binascii
import calendar
import datetime
import os
import struct
import threading
import time
from random import SystemRandom
from typing import Any, NoReturn, Optional, Type, Union
from bson.errors import InvalidId
from bson.tz_util import utc
def _random_bytes() -> bytes:
    """Get the 5-byte random field of an ObjectId."""
    return os.urandom(5)