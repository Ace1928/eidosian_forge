import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
class MetadataFlightServer(FlightServerBase):
    """A Flight server that numbers incoming/outgoing data."""

    def __init__(self, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options

    def do_get(self, context, ticket):
        data = [pa.array([-10, -5, 0, 5, 10])]
        table = pa.Table.from_arrays(data, names=['a'])
        return flight.GeneratorStream(table.schema, self.number_batches(table), options=self.options)

    def do_put(self, context, descriptor, reader, writer):
        counter = 0
        expected_data = [-10, -5, 0, 5, 10]
        while True:
            try:
                batch, buf = reader.read_chunk()
                assert batch.equals(pa.RecordBatch.from_arrays([pa.array([expected_data[counter]])], ['a']))
                assert buf is not None
                client_counter, = struct.unpack('<i', buf.to_pybytes())
                assert counter == client_counter
                writer.write(struct.pack('<i', counter))
                counter += 1
            except StopIteration:
                return

    @staticmethod
    def number_batches(table):
        for idx, batch in enumerate(table.to_batches()):
            buf = struct.pack('<i', idx)
            yield (batch, buf)