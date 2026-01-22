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
class ExchangeFlightServer(FlightServerBase):
    """A server for testing DoExchange."""

    def __init__(self, options=None, **kwargs):
        super().__init__(**kwargs)
        self.options = options

    def do_exchange(self, context, descriptor, reader, writer):
        if descriptor.descriptor_type != flight.DescriptorType.CMD:
            raise pa.ArrowInvalid('Must provide a command descriptor')
        elif descriptor.command == b'echo':
            return self.exchange_echo(context, reader, writer)
        elif descriptor.command == b'get':
            return self.exchange_do_get(context, reader, writer)
        elif descriptor.command == b'put':
            return self.exchange_do_put(context, reader, writer)
        elif descriptor.command == b'transform':
            return self.exchange_transform(context, reader, writer)
        else:
            raise pa.ArrowInvalid('Unknown command: {}'.format(descriptor.command))

    def exchange_do_get(self, context, reader, writer):
        """Emulate DoGet with DoExchange."""
        data = pa.Table.from_arrays([pa.array(range(0, 10 * 1024))], names=['a'])
        writer.begin(data.schema)
        writer.write_table(data)

    def exchange_do_put(self, context, reader, writer):
        """Emulate DoPut with DoExchange."""
        num_batches = 0
        for chunk in reader:
            if not chunk.data:
                raise pa.ArrowInvalid('All chunks must have data.')
            num_batches += 1
        writer.write_metadata(str(num_batches).encode('utf-8'))

    def exchange_echo(self, context, reader, writer):
        """Run a simple echo server."""
        started = False
        for chunk in reader:
            if not started and chunk.data:
                writer.begin(chunk.data.schema, options=self.options)
                started = True
            if chunk.app_metadata and chunk.data:
                writer.write_with_metadata(chunk.data, chunk.app_metadata)
            elif chunk.app_metadata:
                writer.write_metadata(chunk.app_metadata)
            elif chunk.data:
                writer.write_batch(chunk.data)
            else:
                assert False, 'Should not happen'

    def exchange_transform(self, context, reader, writer):
        """Sum rows in an uploaded table."""
        for field in reader.schema:
            if not pa.types.is_integer(field.type):
                raise pa.ArrowInvalid('Invalid field: ' + repr(field))
        table = reader.read_all()
        sums = [0] * table.num_rows
        for column in table:
            for row, value in enumerate(column):
                sums[row] += value.as_py()
        result = pa.Table.from_arrays([pa.array(sums)], names=['sum'])
        writer.begin(result.schema)
        writer.write_table(result)