import os
import sys
from enum import Enum, _simple_enum
@property
def time_mid(self):
    return self.int >> 80 & 65535