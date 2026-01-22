import _imp
import _io
import sys
import _warnings
import marshal
def _calc_mode(path):
    """Calculate the mode permissions for a bytecode file."""
    try:
        mode = _path_stat(path).st_mode
    except OSError:
        mode = 438
    mode |= 128
    return mode