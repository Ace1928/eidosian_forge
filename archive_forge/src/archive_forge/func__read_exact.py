import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _read_exact(fp, n):
    """Read exactly *n* bytes from `fp`

    This method is required because fp may be unbuffered,
    i.e. return short reads.
    """
    data = fp.read(n)
    while len(data) < n:
        b = fp.read(n - len(data))
        if not b:
            raise EOFError('Compressed file ended before the end-of-stream marker was reached')
        data += b
    return data