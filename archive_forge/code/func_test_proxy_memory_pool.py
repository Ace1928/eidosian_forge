import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_proxy_memory_pool():
    pool = pa.proxy_memory_pool(pa.default_memory_pool())
    check_allocated_bytes(pool)
    wr = weakref.ref(pool)
    assert wr() is not None
    del pool
    assert wr() is None