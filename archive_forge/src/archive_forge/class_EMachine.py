import enum
import os
import struct
from typing import IO, Optional, Tuple
class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArc64 = 183