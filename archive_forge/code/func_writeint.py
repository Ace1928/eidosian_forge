import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
def writeint(fd, n, pos=None):
    """Write 64 bit integer n at pos or current position."""
    if pos is not None:
        fd.seek(pos)
    a = np.array(n, np.int64)
    if not np.little_endian:
        a.byteswap(True)
    fd.write(a.tobytes())