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
@pytest.mark.requires_testing_data
def test_tls_disable_server_verification():
    """Try a simple do_get call over TLS with server verification disabled."""
    table = simple_ints_table()
    certs = example_tls_certs()
    with ConstantFlightServer(tls_certificates=certs['certificates']) as s:
        try:
            client = FlightClient(('localhost', s.port), disable_server_verification=True)
        except NotImplementedError:
            pytest.skip('disable_server_verification feature is not available')
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)
        client.close()