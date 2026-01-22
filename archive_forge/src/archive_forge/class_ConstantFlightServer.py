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
class ConstantFlightServer(FlightServerBase):
    """A Flight server that always returns the same data.

    See ARROW-4796: this server implementation will segfault if Flight
    does not properly hold a reference to the Table object.
    """
    CRITERIA = b'the expected criteria'

    def __init__(self, location=None, options=None, **kwargs):
        super().__init__(location, **kwargs)
        self.table_factories = {b'ints': simple_ints_table, b'dicts': simple_dicts_table, b'multi': multiple_column_table}
        self.options = options

    def list_flights(self, context, criteria):
        if criteria == self.CRITERIA:
            yield flight.FlightInfo(pa.schema([]), flight.FlightDescriptor.for_path('/foo'), [], -1, -1)

    def do_get(self, context, ticket):
        table = self.table_factories[ticket.ticket]()
        return flight.RecordBatchStream(table, options=self.options)