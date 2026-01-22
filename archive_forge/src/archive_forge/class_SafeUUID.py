import os
import sys
from enum import Enum, _simple_enum
@_simple_enum(Enum)
class SafeUUID:
    safe = 0
    unsafe = -1
    unknown = None