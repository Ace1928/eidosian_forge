import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
@contextlib.contextmanager
def allocate_bytes(pool, nbytes):
    """
    Temporarily allocate *nbytes* from the given *pool*.
    """
    arr = pa.array([b'x' * nbytes], type=pa.binary(), memory_pool=pool)
    buf = arr.buffers()[2]
    arr = None
    assert len(buf) == nbytes
    try:
        yield
    finally:
        buf = None