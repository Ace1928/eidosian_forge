from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
class IpcFixture:
    write_stats = None

    def __init__(self, sink_factory=lambda: io.BytesIO()):
        self._sink_factory = sink_factory
        self.sink = self.get_sink()

    def get_sink(self):
        return self._sink_factory()

    def get_source(self):
        return self.sink.getvalue()

    def write_batches(self, num_batches=5, as_table=False):
        nrows = 5
        schema = pa.schema([('one', pa.float64()), ('two', pa.utf8())])
        writer = self._get_writer(self.sink, schema)
        batches = []
        for i in range(num_batches):
            batch = pa.record_batch([np.random.randn(nrows), ['foo', None, 'bar', 'bazbaz', 'qux']], schema=schema)
            batches.append(batch)
        if as_table:
            table = pa.Table.from_batches(batches)
            writer.write_table(table)
        else:
            for batch in batches:
                writer.write_batch(batch)
        self.write_stats = writer.stats
        writer.close()
        return batches