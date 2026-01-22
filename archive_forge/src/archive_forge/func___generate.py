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
def __generate(self) -> None:
    """Generate a new value for this ObjectId."""
    oid = struct.pack('>I', int(time.time()))
    oid += ObjectId._random()
    with ObjectId._inc_lock:
        oid += struct.pack('>I', ObjectId._inc)[1:4]
        ObjectId._inc = (ObjectId._inc + 1) % (_MAX_COUNTER_VALUE + 1)
    self.__id = oid