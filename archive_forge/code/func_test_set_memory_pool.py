import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_set_memory_pool():
    old_pool = pa.default_memory_pool()
    pool = pa.proxy_memory_pool(old_pool)
    pa.set_memory_pool(pool)
    try:
        allocated_before = pool.bytes_allocated()
        with allocate_bytes(None, 512):
            assert pool.bytes_allocated() == allocated_before + 512
        assert pool.bytes_allocated() == allocated_before
    finally:
        pa.set_memory_pool(old_pool)