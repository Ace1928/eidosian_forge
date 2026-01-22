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
def _raise_invalid_id(oid: str) -> NoReturn:
    raise InvalidId('%r is not a valid ObjectId, it must be a 12-byte input or a 24-character hex string' % oid)