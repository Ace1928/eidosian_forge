import struct
import sys
import numpy as np
def unstr64(i):
    """Convert an int64 to a string."""
    b = struct.pack('@q', i)
    return b.decode('ascii').strip('\x00')