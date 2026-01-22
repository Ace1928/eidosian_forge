import json
import mmap
import os
import struct
from typing import List
def _pack_two_doubles(data, pos, value, timestamp):
    data[pos:pos + 16] = _pack_two_doubles_func(value, timestamp)