import os
import sys
from enum import Enum, _simple_enum
@property
def time_hi_version(self):
    return self.int >> 64 & 65535