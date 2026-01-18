from __future__ import absolute_import
import itertools
import sys
from struct import pack
def write_to_stdout(data):
    """Writes bytes to stdout

    :type data: bytes
    """
    if PY2:
        sys.stdout.write(data)
    else:
        sys.stdout.buffer.write(data)