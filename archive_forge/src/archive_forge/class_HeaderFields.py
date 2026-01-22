from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class HeaderFields(IntEnum):
    path = 1
    interface = 2
    member = 3
    error_name = 4
    reply_serial = 5
    destination = 6
    sender = 7
    signature = 8
    unix_fds = 9