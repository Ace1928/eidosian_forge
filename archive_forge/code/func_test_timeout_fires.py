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
def test_timeout_fires():
    """Make sure timeouts fire on slow requests."""
    with SlowFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        action = flight.Action('', b'')
        options = flight.FlightCallOptions(timeout=0.2)
        with pytest.raises(flight.FlightTimedOutError):
            list(client.do_action(action, options=options))