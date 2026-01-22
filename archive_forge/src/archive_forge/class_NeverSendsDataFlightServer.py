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
class NeverSendsDataFlightServer(FlightServerBase):
    """A Flight server that never actually yields data."""
    schema = pa.schema([('a', pa.int32())])

    def do_get(self, context, ticket):
        if ticket.ticket == b'yield_data':
            data = [self.schema.empty_table(), self.schema.empty_table(), pa.RecordBatch.from_arrays([range(5)], schema=self.schema)]
            return flight.GeneratorStream(self.schema, data)
        return flight.GeneratorStream(self.schema, itertools.repeat(self.schema.empty_table()))