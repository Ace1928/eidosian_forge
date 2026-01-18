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
def test_roundtrip_types():
    """Make sure serializable types round-trip."""
    action = flight.Action('action1', b'action1-body')
    assert action == flight.Action.deserialize(action.serialize())
    ticket = flight.Ticket('foo')
    assert ticket == flight.Ticket.deserialize(ticket.serialize())
    result = flight.Result(b'result1')
    assert result == flight.Result.deserialize(result.serialize())
    basic_auth = flight.BasicAuth('username1', 'password1')
    assert basic_auth == flight.BasicAuth.deserialize(basic_auth.serialize())
    schema_result = flight.SchemaResult(pa.schema([('a', pa.int32())]))
    assert schema_result == flight.SchemaResult.deserialize(schema_result.serialize())
    desc = flight.FlightDescriptor.for_command('test')
    assert desc == flight.FlightDescriptor.deserialize(desc.serialize())
    desc = flight.FlightDescriptor.for_path('a', 'b', 'test.arrow')
    assert desc == flight.FlightDescriptor.deserialize(desc.serialize())
    info = flight.FlightInfo(pa.schema([('a', pa.int32())]), desc, [flight.FlightEndpoint(b'', ['grpc://test']), flight.FlightEndpoint(b'', [flight.Location.for_grpc_tcp('localhost', 5005)])], -1, -1)
    info2 = flight.FlightInfo.deserialize(info.serialize())
    assert info.schema == info2.schema
    assert info.descriptor == info2.descriptor
    assert info.total_bytes == info2.total_bytes
    assert info.total_records == info2.total_records
    assert info.endpoints == info2.endpoints
    endpoint = flight.FlightEndpoint(ticket, ['grpc://test', flight.Location.for_grpc_tcp('localhost', 5005)])
    assert endpoint == flight.FlightEndpoint.deserialize(endpoint.serialize())