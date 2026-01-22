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
class GetInfoFlightServer(FlightServerBase):
    """A Flight server that tests GetFlightInfo."""

    def get_flight_info(self, context, descriptor):
        return flight.FlightInfo(pa.schema([('a', pa.int32())]), descriptor, [flight.FlightEndpoint(b'', ['grpc://test']), flight.FlightEndpoint(b'', [flight.Location.for_grpc_tcp('localhost', 5005)])], -1, -1)

    def get_schema(self, context, descriptor):
        info = self.get_flight_info(context, descriptor)
        return flight.SchemaResult(info.schema)