import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
def test_write_dataset_with_backpressure(tempdir):
    consumer_gate = threading.Event()

    class GatingFs(ProxyHandler):

        def open_output_stream(self, path, metadata):
            consumer_gate.wait()
            return self._fs.open_output_stream(path, metadata=metadata)
    gating_fs = fs.PyFileSystem(GatingFs(fs.LocalFileSystem()))
    schema = pa.schema([pa.field('data', pa.int32())])
    batch = pa.record_batch([pa.array(list(range(1000000)))], schema=schema)
    batches_read = 0
    min_backpressure = 32
    end = 200
    keep_going = True

    def counting_generator():
        nonlocal batches_read
        while batches_read < end:
            if not keep_going:
                return
            time.sleep(0.01)
            batches_read += 1
            yield batch
    scanner = ds.Scanner.from_batches(counting_generator(), schema=schema, use_threads=True)
    write_thread = threading.Thread(target=lambda: ds.write_dataset(scanner, str(tempdir), format='parquet', filesystem=gating_fs))
    write_thread.start()
    try:
        start = time.time()

        def duration():
            return time.time() - start
        last_value = 0
        backpressure_probably_hit = False
        while duration() < 10:
            if batches_read > min_backpressure:
                if batches_read == last_value:
                    backpressure_probably_hit = True
                    break
                last_value = batches_read
            time.sleep(0.5)
        assert backpressure_probably_hit
    finally:
        keep_going = False
        consumer_gate.set()
        write_thread.join()