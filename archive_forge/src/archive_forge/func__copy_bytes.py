from the actual Python version.
import sys
import abc
def _copy_bytes(start, end, seq):
    """Return an immutable copy of a sequence (byte string, byte array, memoryview)
    in a certain interval [start:seq]"""
    if isinstance(seq, memoryview):
        return seq[start:end].tobytes()
    elif isinstance(seq, bytearray):
        return bytes(seq[start:end])
    else:
        return seq[start:end]