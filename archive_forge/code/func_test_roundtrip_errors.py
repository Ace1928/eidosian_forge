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
def test_roundtrip_errors():
    """Ensure that Flight errors propagate from server to client."""
    with ErrorFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        for arg, exc_type in ErrorFlightServer.error_cases().items():
            with pytest.raises(exc_type, match='.*foo.*'):
                list(client.do_action(flight.Action(arg, b'')))
        with pytest.raises(flight.FlightInternalError, match='.*foo.*'):
            list(client.list_flights())
        data = [pa.array([-10, -5, 0, 5, 10])]
        table = pa.Table.from_arrays(data, names=['a'])
        exceptions = {'internal': flight.FlightInternalError, 'timedout': flight.FlightTimedOutError, 'cancel': flight.FlightCancelledError, 'unauthenticated': flight.FlightUnauthenticatedError, 'unauthorized': flight.FlightUnauthorizedError}
        for command, exception in exceptions.items():
            with pytest.raises(exception, match='.*foo.*'):
                writer, reader = client.do_put(flight.FlightDescriptor.for_command(command), table.schema)
                writer.write_table(table)
                writer.close()
            with pytest.raises(exception, match='.*foo.*'):
                writer, reader = client.do_put(flight.FlightDescriptor.for_command(command), table.schema)
                writer.close()