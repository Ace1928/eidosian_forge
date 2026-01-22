import _imp
import _io
import sys
import _warnings
import marshal
def _pack_uint32(x):
    """Convert a 32-bit integer to little-endian."""
    return (int(x) & 4294967295).to_bytes(4, 'little')