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
def test_server_middleware_same_thread():
    """Ensure that server middleware run on the same thread as the RPC."""
    with HeaderFlightServer(middleware={'test': HeaderServerMiddlewareFactory()}) as server, FlightClient(('localhost', server.port)) as client:
        results = list(client.do_action(flight.Action(b'test', b'')))
        assert len(results) == 1
        value = results[0].body.to_pybytes()
        assert b'right value' == value