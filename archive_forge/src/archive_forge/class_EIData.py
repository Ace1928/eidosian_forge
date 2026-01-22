import enum
import os
import struct
from typing import IO, Optional, Tuple
class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2