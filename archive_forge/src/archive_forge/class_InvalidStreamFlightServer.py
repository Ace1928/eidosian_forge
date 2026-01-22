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
class InvalidStreamFlightServer(FlightServerBase):
    """A Flight server that tries to return messages with differing schemas."""
    schema = pa.schema([('a', pa.int32())])

    def do_get(self, context, ticket):
        data1 = [pa.array([-10, -5, 0, 5, 10], type=pa.int32())]
        data2 = [pa.array([-10.0, -5.0, 0.0, 5.0, 10.0], type=pa.float64())]
        assert data1.type != data2.type
        table1 = pa.Table.from_arrays(data1, names=['a'])
        table2 = pa.Table.from_arrays(data2, names=['a'])
        assert table1.schema == self.schema
        return flight.GeneratorStream(self.schema, [table1, table2])