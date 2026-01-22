import sys
import os
from pathlib import Path
import io
def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')